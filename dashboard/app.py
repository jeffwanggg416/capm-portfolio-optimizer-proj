import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import numpy as np
import pandas as pd

from src.validation import build_universe
from src.data_loader import get_cached_or_fresh_prices, fetch_risk_free_rate
from src import database
from src.returns import build_returns_dataframe
from src.covariance import calculate_covariance_matrix
from src.capm import build_capm_summary, calculate_capm_expected_returns
from src.optimizer import minimum_variance_portfolio, maximum_sharpe_portfolio, target_risk_portfolio
from src.risk import summarize_portfolio, build_weight_shift_table
from src.visualization import generate_random_portfolios, plot_efficient_frontier, plot_weight_shift
from config import MARKET_TICKER

st.set_page_config(page_title="CAPM Portfolio Optimizer", layout="wide")

# ---------------------------------------------------------------
# Custom styling: rounded pill input/button, metric tiles,
# bar-list allocations, accent border on the highlighted card,
# and a quiet disclaimer section.
# ---------------------------------------------------------------
st.markdown("""
<style>
    div[data-testid="stTextInput"] input {
        border-radius: 22px !important;
        height: 44px !important;
        padding-left: 18px !important;
        text-align: center;
    }
    div[data-testid="stButton"] button {
        border-radius: 22px !important;
        height: 44px !important;
    }
    .metric-tile {
        background: rgba(120,120,120,0.08);
        border-radius: 8px;
        padding: 10px;
        text-align: left;
    }
    .metric-tile p.label {
        font-size: 12px;
        color: rgba(120,120,120,0.9);
        margin: 0 0 2px 0;
    }
    .metric-tile p.value {
        font-size: 17px;
        font-weight: 600;
        margin: 0;
    }
    .portfolio-card {
        background: rgba(120,120,120,0.03);
        border-radius: 12px;
        border: 0.5px solid rgba(120,120,120,0.25);
        padding: 1rem 1.25rem;
        margin-bottom: 1rem;
    }
    .portfolio-card.highlight {
        border: 2px solid #378ADD;
    }
    .badge {
        display: inline-block;
        background: rgba(55,138,221,0.15);
        color: #185FA5;
        font-size: 12px;
        padding: 3px 10px;
        border-radius: 8px;
        margin-bottom: 8px;
    }
    .section-label {
        font-size: 13px;
        color: rgba(120,120,120,0.9);
        text-transform: uppercase;
        letter-spacing: 0.02em;
        margin-bottom: 6px;
    }
    .bar-row-label {
        display: flex;
        justify-content: space-between;
        font-size: 13px;
        margin-bottom: 4px;
    }
    .bar-track {
        height: 6px;
        background: rgba(120,120,120,0.12);
        border-radius: 3px;
        margin-bottom: 8px;
    }
    .bar-fill {
        height: 100%;
        border-radius: 3px;
        background: #378ADD;
    }
    .disclaimer {
        font-size: 12px;
        color: rgba(120,120,120,0.9);
        line-height: 1.6;
        border-top: 0.5px solid rgba(120,120,120,0.25);
        padding-top: 1rem;
        margin-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data(ttl=3600)
def load_data(tickers: list) -> pd.DataFrame:
    for t in tickers:
        get_cached_or_fresh_prices(t)
    price_data = {t: database.load_price_history(t) for t in tickers}
    return build_returns_dataframe(price_data)


@st.cache_data(ttl=3600)
def get_risk_free_rate() -> float:
    return fetch_risk_free_rate()


def metric_tile_html(label, value):
    return f'<div class="metric-tile"><p class="label">{label}</p><p class="value">{value}</p></div>'


def bar_row_html(ticker, pct):
    return (
        f'<div class="bar-row-label"><span>{ticker}</span>'
        f'<span style="color:rgba(120,120,120,0.9);">{pct:.0f}%</span></div>'
        f'<div class="bar-track"><div class="bar-fill" style="width:{max(pct, 0.5)}%;"></div></div>'
    )


def render_portfolio_card(title, icon_label, weights, summary, all_tickers, highlight=False, badge_text=None):
    card_class = "portfolio-card highlight" if highlight else "portfolio-card"
    badge_html = f'<span class="badge">{badge_text}</span>' if badge_text else ""

    tiles = "".join([
        metric_tile_html("Return", f"{summary['expected_return']*100:.1f}%"),
        metric_tile_html("Volatility", f"{summary['volatility']*100:.1f}%"),
        metric_tile_html("Sharpe", f"{summary['sharpe_ratio']:.2f}"),
        metric_tile_html("Beta", f"{summary['beta']:.2f}"),
    ])

    full_weights = weights.reindex(all_tickers).fillna(0).sort_values(ascending=False)
    bars = "".join([bar_row_html(ticker, pct * 100) for ticker, pct in full_weights.items()])

    html = (
        f'<div class="{card_class}">'
        f'{badge_html}'
        f'<p style="font-weight:600; font-size:15px; margin:0 0 12px 0;">{icon_label} {title}</p>'
        f'<div style="display:grid; grid-template-columns:repeat(4,1fr); gap:10px; margin-bottom:12px;">{tiles}</div>'
        f'<div style="border-top:0.5px solid rgba(120,120,120,0.25); padding-top:10px;">{bars}</div>'
        f'</div>'
    )
    st.markdown(html, unsafe_allow_html=True)


# ---------------------------------------------------------------
# Header + centered ticker input
# ---------------------------------------------------------------
st.markdown("<h2 style='text-align:center; margin-bottom:2px;'>CAPM portfolio optimizer</h2>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align:center; font-size:13px; color:rgba(120,120,120,0.9); margin-bottom:16px;'>"
    "Enter tickers to build minimum variance, maximum Sharpe, and target risk portfolios.</p>",
    unsafe_allow_html=True,
)

input_col_l, input_col_mid, input_col_r = st.columns([1, 2, 1])
with input_col_mid:
    text_col, button_col = st.columns([3, 1])
    with text_col:
        raw_input = st.text_input(
            "Tickers", value="AAPL, MSFT, NVDA, JPM, COST",
            label_visibility="collapsed", placeholder="AAPL, MSFT, NVDA, JPM, COST",
        )
    with button_col:
        run_button = st.button("Run", use_container_width=True)

st.markdown(
    "<p style='text-align:center; font-size:12px; color:rgba(120,120,120,0.7); margin-top:4px;'>"
    "VTIP is added automatically as a defensive holding.</p>",
    unsafe_allow_html=True,
)

target_vol_pct = st.session_state.get("target_vol_pct", 15)

if run_button:
    try:
        asset_tickers = build_universe(raw_input)
    except ValueError as e:
        st.error(str(e))
        st.stop()

    st.session_state["asset_tickers"] = asset_tickers
    st.session_state["has_run"] = True

if st.session_state.get("has_run"):
    asset_tickers = st.session_state["asset_tickers"]
    all_tickers = asset_tickers + [MARKET_TICKER]

    with st.spinner("Fetching data and running optimizations..."):
        returns_df = load_data(all_tickers)
        rf = get_risk_free_rate()

        capm_summary = build_capm_summary(returns_df, MARKET_TICKER, rf)
        capm_summary = calculate_capm_expected_returns(capm_summary, returns_df[MARKET_TICKER], rf)

        cov_matrix = calculate_covariance_matrix(returns_df[asset_tickers])
        expected_returns = capm_summary["capm_expected_return"]
        betas = capm_summary["beta"]

        gmv_result = minimum_variance_portfolio(cov_matrix)
        sharpe_result = maximum_sharpe_portfolio(expected_returns, cov_matrix, rf)

        gmv_summary = summarize_portfolio(gmv_result["weights"], expected_returns, betas, cov_matrix, rf)
        sharpe_summary = summarize_portfolio(sharpe_result["weights"], expected_returns, betas, cov_matrix, rf)

    # --- Fixed portfolios: two cards side by side ---
    st.markdown("<p class='section-label'>Fixed portfolios</p>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        render_portfolio_card("Minimum variance", "🛡", gmv_result["weights"], gmv_summary, asset_tickers)
    with col2:
        render_portfolio_card(
            "Maximum Sharpe", "📈", sharpe_result["weights"], sharpe_summary, asset_tickers,
            highlight=True, badge_text="Best Sharpe",
        )

    # --- Target Risk: its own full-width section with the slider ---
    st.markdown("<p class='section-label'>Target risk — adjustable</p>", unsafe_allow_html=True)

    target_vol_pct = st.slider(
        "Target annualized volatility (%)", 8, 30, target_vol_pct, key="target_vol_pct",
    )

    target_result = target_risk_portfolio(expected_returns, cov_matrix, target_vol_pct / 100)
    if not target_result["feasible"]:
        st.warning(f"{target_vol_pct}% is below the achievable minimum. Showing Minimum Variance instead.")
    target_summary = summarize_portfolio(target_result["weights"], expected_returns, betas, cov_matrix, rf)

    render_portfolio_card(f"Target risk ({target_vol_pct}%)", "🎯", target_result["weights"], target_summary, asset_tickers)

    # --- Chart tabs ---
    tab1, tab2, tab3, tab4 = st.tabs(["Efficient frontier", "CAPM summary", "Weight shift", "Backtest"])

    with tab1:
        random_portfolios = generate_random_portfolios(expected_returns, cov_matrix, n_portfolios=3000)

        gmv_point = {"volatility": gmv_summary["volatility"], "expected_return": gmv_summary["expected_return"]}
        sharpe_point = {"volatility": sharpe_summary["volatility"], "expected_return": sharpe_summary["expected_return"]}
        target_point = {"volatility": target_summary["volatility"], "expected_return": target_summary["expected_return"]}

        frontier_targets = np.linspace(gmv_result["volatility"], 0.35, 50)
        frontier_results = []
        for t in frontier_targets:
            r = target_risk_portfolio(expected_returns, cov_matrix, t)
            if r["feasible"]:
                frontier_results.append({"volatility": r["volatility"], "return": r["expected_return"]})
        frontier_points = pd.DataFrame(frontier_results)

        fig1 = plot_efficient_frontier(random_portfolios, frontier_points, gmv_point, sharpe_point, target_point, rf)
        st.pyplot(fig1)

    with tab2:
        st.dataframe(capm_summary.round(4), use_container_width=True)

    with tab3:
        volatility_targets = [0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20, 0.22, 0.25, 0.30]
        shift_table = build_weight_shift_table(expected_returns, cov_matrix, volatility_targets)
        fig2 = plot_weight_shift(shift_table)
        st.pyplot(fig2)

    with tab4:
        st.info("Backtest results go here (Phase 17 engine — wire in next).")

    # --- Disclaimer ---
    st.markdown("""
    <div class="disclaimer">
        <p><strong>About VTIP:</strong> VTIP (Vanguard Short-Term Inflation-Protected Securities ETF) holds
        short-duration U.S. Treasury Inflation-Protected Securities and is included as a defensive,
        low-volatility holding. It is treated as an ordinary investable asset the optimizer can allocate
        0-100% to, not as a stand-in for the risk-free rate.</p>
        <p><strong>About the risk-free rate:</strong> Rf is pulled live from the 3-month U.S. Treasury bill
        yield (ticker ^IRX) and used in the CAPM and Sharpe ratio calculations. This is a separate,
        theoretical figure from VTIP's own realized return.</p>
    </div>
    """, unsafe_allow_html=True)

else:
    st.info("Enter tickers above and click Run to see all three portfolios.")
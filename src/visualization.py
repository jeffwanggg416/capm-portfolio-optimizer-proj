import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline

from src.covariance import portfolio_volatility

ACCENT_BLUE = "#378ADD"
MUTED_GRAY = "#888780"
TEAL = "#1D9E75"
CORAL = "#D85A30"
PURPLE = "#7F77DD"


def generate_random_portfolios(
    expected_returns: pd.Series,
    cov_matrix: pd.DataFrame,
    n_portfolios: int = 5000,
) -> pd.DataFrame:
    tickers = cov_matrix.columns.tolist()
    n_assets = len(tickers)
    exp_returns_array = expected_returns.loc[tickers].values

    results = []

    for _ in range(n_portfolios):
        raw_weights = np.random.random(n_assets)
        weights = raw_weights / raw_weights.sum()

        port_return = np.dot(weights, exp_returns_array)
        port_vol = portfolio_volatility(weights, cov_matrix)

        results.append({"return": port_return, "volatility": port_vol})

    return pd.DataFrame(results)


def _apply_theme(ax):
    ax.set_facecolor("none")
    ax.tick_params(colors=MUTED_GRAY, labelsize=9)
    for spine in ax.spines.values():
        spine.set_color(MUTED_GRAY)
        spine.set_linewidth(0.5)
        spine.set_alpha(0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.xaxis.label.set_color(MUTED_GRAY)
    ax.yaxis.label.set_color(MUTED_GRAY)
    ax.grid(alpha=0.15, color=MUTED_GRAY, linewidth=0.5)


def plot_efficient_frontier(
    random_portfolios: pd.DataFrame,
    frontier_points: pd.DataFrame,
    gmv_point: dict,
    sharpe_point: dict,
    target_point: dict,
    risk_free_rate: float,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=180)
    fig.patch.set_alpha(0)

    ax.scatter(
        random_portfolios["volatility"], random_portfolios["return"],
        c=MUTED_GRAY, s=5, alpha=0.2, label="Random portfolios", linewidths=0,
    )

    frontier_sorted = frontier_points.sort_values("volatility").drop_duplicates(subset="volatility")

    x = frontier_sorted["volatility"].values
    y = frontier_sorted["return"].values
    if len(x) >= 4:
        x_smooth = np.linspace(x.min(), x.max(), 300)
        spline = make_interp_spline(x, y, k=3)
        y_smooth = spline(x_smooth)
    else:
        x_smooth, y_smooth = x, y

    ax.plot(x_smooth, y_smooth, color=ACCENT_BLUE, linewidth=2, label="Efficient frontier")

    ax.scatter(gmv_point["volatility"], gmv_point["expected_return"],
               color=TEAL, marker="o", s=70, label="Minimum variance", zorder=5, edgecolors="none")
    ax.scatter(sharpe_point["volatility"], sharpe_point["expected_return"],
               color=CORAL, marker="o", s=70, label="Maximum Sharpe", zorder=5, edgecolors="none")
    ax.scatter(target_point["volatility"], target_point["expected_return"],
               color=PURPLE, marker="o", s=70, label="Target risk", zorder=5, edgecolors="none")

    cml_x = np.linspace(0, x.max(), 100)
    cml_slope = (sharpe_point["expected_return"] - risk_free_rate) / sharpe_point["volatility"]
    cml_y = risk_free_rate + cml_slope * cml_x
    ax.plot(cml_x, cml_y, color=MUTED_GRAY, linestyle="--", linewidth=1, alpha=0.5,
            label="Capital market line (theoretical)")

    ax.set_xlabel("Annualized volatility", fontsize=10)
    ax.set_ylabel("Expected return (CAPM)", fontsize=10)
    _apply_theme(ax)

    ax.legend(loc="best", fontsize=8, frameon=False, labelcolor=MUTED_GRAY)
    fig.tight_layout()

    return fig


def plot_weight_shift(shift_table: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 4), dpi=180)
    fig.patch.set_alpha(0)

    palette = [ACCENT_BLUE, TEAL, CORAL, PURPLE, "#D4537E", "#BA7517"]
    x_labels = shift_table.columns.tolist()
    x_positions = range(len(x_labels))

    ax.stackplot(
        x_positions, shift_table.values,
        labels=shift_table.index.tolist(), colors=palette[:len(shift_table)], alpha=0.85,
    )

    ax.set_xticks(list(x_positions))
    ax.set_xticklabels(x_labels, fontsize=9)
    ax.set_xlabel("Target volatility", fontsize=10)
    ax.set_ylabel("Portfolio weight", fontsize=10)
    ax.set_ylim(0, 1)
    _apply_theme(ax)

    ax.legend(loc="upper left", fontsize=8, bbox_to_anchor=(1.01, 1), frameon=False, labelcolor=MUTED_GRAY)
    fig.tight_layout()

    return fig


if __name__ == "__main__":
    from src import database
    from src.returns import build_returns_dataframe
    from src.covariance import calculate_covariance_matrix
    from src.capm import build_capm_summary, calculate_capm_expected_returns
    from src.data_loader import fetch_risk_free_rate
    from src.optimizer import minimum_variance_portfolio, maximum_sharpe_portfolio, target_risk_portfolio
    from src.risk import build_weight_shift_table
    from config import MARKET_TICKER

    all_tickers = ["AAPL", "MSFT", "NVDA", "VTIP", "SPY"]
    price_data = {t: database.load_price_history(t) for t in all_tickers}
    returns_df = build_returns_dataframe(price_data)

    rf = fetch_risk_free_rate()
    capm_summary = build_capm_summary(returns_df, MARKET_TICKER, rf)
    capm_summary = calculate_capm_expected_returns(capm_summary, returns_df[MARKET_TICKER], rf)

    asset_tickers = ["AAPL", "MSFT", "NVDA", "VTIP"]
    cov_matrix = calculate_covariance_matrix(returns_df[asset_tickers])
    expected_returns = capm_summary["capm_expected_return"]

    gmv = minimum_variance_portfolio(cov_matrix)
    sharpe_port = maximum_sharpe_portfolio(expected_returns, cov_matrix, rf)
    target_port = target_risk_portfolio(expected_returns, cov_matrix, 0.15)

    gmv_point = {
        "volatility": gmv["volatility"],
        "expected_return": np.dot(gmv["weights"].values, expected_returns.loc[asset_tickers].values),
    }
    sharpe_point = {"volatility": sharpe_port["volatility"], "expected_return": sharpe_port["expected_return"]}
    target_point = {"volatility": target_port["volatility"], "expected_return": target_port["expected_return"]}

    random_portfolios = generate_random_portfolios(expected_returns, cov_matrix, n_portfolios=5000)

    frontier_targets = np.linspace(gmv["volatility"], 0.35, 50)
    frontier_results = []
    for t in frontier_targets:
        r = target_risk_portfolio(expected_returns, cov_matrix, t)
        if r["feasible"]:
            frontier_results.append({"volatility": r["volatility"], "return": r["expected_return"]})
    frontier_points = pd.DataFrame(frontier_results)

    fig1 = plot_efficient_frontier(random_portfolios, frontier_points, gmv_point, sharpe_point, target_point, rf)
    fig1.savefig("efficient_frontier_test.png", dpi=180, bbox_inches="tight")
    print("Saved chart to efficient_frontier_test.png")

    volatility_targets = [0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20, 0.22, 0.25, 0.30]
    shift_table = build_weight_shift_table(expected_returns, cov_matrix, volatility_targets)

    fig2 = plot_weight_shift(shift_table)
    fig2.savefig("weight_shift_test.png", dpi=180, bbox_inches="tight")
    print("Saved chart to weight_shift_test.png")
import pandas as pd
import statsmodels.api as sm

from src.returns import annualize_return, annualize_volatility


def run_capm_regression(
    security_returns: pd.Series,
    market_returns: pd.Series,
    risk_free_rate: float,
) -> dict:
    """
    Run a CAPM regression for a single security against the market.

    Ri - Rf = alpha + beta(Rm - Rf) + epsilon

    Parameters
    ----------
    security_returns : pd.Series
        Daily simple returns for the security.
    market_returns : pd.Series
        Daily simple returns for the market benchmark (SPY).
    risk_free_rate : float
        Annualized risk-free rate as a decimal (e.g. 0.04).

    Returns
    -------
    dict with keys: beta, alpha, r_squared
    """
    daily_rf = risk_free_rate / 252

    excess_security = security_returns - daily_rf
    excess_market = market_returns - daily_rf

    X = sm.add_constant(excess_market)
    y = excess_security

    model = sm.OLS(y, X).fit()

    alpha = model.params.iloc[0]
    beta = model.params.iloc[1]
    r_squared = model.rsquared

    return {
        "beta": beta,
        "alpha": alpha,
        "r_squared": r_squared,
    }


def build_capm_summary(
    returns_df: pd.DataFrame,
    market_ticker: str,
    risk_free_rate: float,
) -> pd.DataFrame:
    """
    Run CAPM regressions for every security in returns_df against
    the market ticker, and combine results with historical return
    and volatility stats into one summary table.

    Parameters
    ----------
    returns_df : pd.DataFrame
        Daily returns, columns are tickers, including the market ticker.
    market_ticker : str
        Which column in returns_df represents the market (e.g. "SPY").
    risk_free_rate : float
        Annualized risk-free rate as a decimal.

    Returns
    -------
    pd.DataFrame indexed by ticker, with columns:
        beta, alpha, r_squared, historical_annual_return,
        historical_annual_volatility
    """
    market_returns = returns_df[market_ticker]
    results = {}

    for ticker in returns_df.columns:
        if ticker == market_ticker:
            continue

        security_returns = returns_df[ticker]
        capm_result = run_capm_regression(security_returns, market_returns, risk_free_rate)

        results[ticker] = {
            "beta": capm_result["beta"],
            "alpha": capm_result["alpha"],
            "r_squared": capm_result["r_squared"],
            "historical_annual_return": annualize_return(security_returns.mean()),
            "historical_annual_volatility": annualize_volatility(security_returns.std()),
        }

    return pd.DataFrame(results).T


def calculate_capm_expected_returns(
    capm_summary: pd.DataFrame,
    market_returns: pd.Series,
    risk_free_rate: float,
) -> pd.DataFrame:
    """
    Add a CAPM expected return column to the capm_summary table.

    E(Ri) = Rf + Beta_i * (E(Rm) - Rf)

    E(Rm) is estimated as the annualized historical mean return
    of the market benchmark over the sample period. This is a
    documented simplification; see README for discussion of
    alternative equity risk premium approaches.

    Parameters
    ----------
    capm_summary : pd.DataFrame
        Output of build_capm_summary(), must have a 'beta' column.
    market_returns : pd.Series
        Daily simple returns for the market benchmark.
    risk_free_rate : float
        Annualized risk-free rate as a decimal.

    Returns
    -------
    pd.DataFrame
        capm_summary with an added 'capm_expected_return' column.
    """
    expected_market_return = annualize_return(market_returns.mean())

    capm_summary = capm_summary.copy()
    capm_summary["capm_expected_return"] = (
        risk_free_rate + capm_summary["beta"] * (expected_market_return - risk_free_rate)
    )

    return capm_summary


if __name__ == "__main__":
    from src import database
    from src.returns import build_returns_dataframe
    from src.data_loader import fetch_risk_free_rate
    from config import MARKET_TICKER

    tickers = ["AAPL", "MSFT", "NVDA", "VTIP", "SPY"]
    price_data = {t: database.load_price_history(t) for t in tickers}
    returns_df = build_returns_dataframe(price_data)

    rf = fetch_risk_free_rate()
    print(f"Risk-free rate: {rf}\n")

    summary = build_capm_summary(returns_df, MARKET_TICKER, rf)
    summary = calculate_capm_expected_returns(summary, returns_df[MARKET_TICKER], rf)
    print(summary)
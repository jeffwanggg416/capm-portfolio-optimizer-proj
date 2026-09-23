import numpy as np
import pandas as pd

from src.covariance import calculate_covariance_matrix
from src.capm import build_capm_summary, calculate_capm_expected_returns
from src.optimizer import minimum_variance_portfolio, maximum_sharpe_portfolio, target_risk_portfolio


def generate_rebalance_dates(returns_df: pd.DataFrame, min_formation_days: int = 252) -> list:
    """
    Generate a list of quarterly rebalance dates for walk-forward backtesting.

    Parameters
    ----------
    returns_df : pd.DataFrame
        Full daily returns history, all tickers, entire dataset.
    min_formation_days : int
        Minimum number of trading days required before the first
        rebalance date, so early CAPM/covariance estimates aren't
        based on too little data. Defaults to ~1 year.

    Returns
    -------
    list of pd.Timestamp
        Dates on which to re-form the portfolio, spaced roughly
        every quarter (63 trading days), starting after the
        minimum formation window.
    """
    all_dates = returns_df.index
    start_index = min_formation_days
    rebalance_indices = range(start_index, len(all_dates), 63)  # ~63 trading days per quarter

    return [all_dates[i] for i in rebalance_indices]


def run_single_period(
    formation_returns: pd.DataFrame,
    test_returns: pd.DataFrame,
    market_ticker: str,
    risk_free_rate: float,
    asset_tickers: list,
    method: str,
    target_volatility: float = None,
) -> pd.Series:
    """
    For one formation/test cycle: estimate CAPM and covariance using
    ONLY formation_returns, determine weights via the specified method,
    then apply those fixed weights to test_returns to get realized
    portfolio returns for the test period.

    Parameters
    ----------
    formation_returns : pd.DataFrame
        Historical returns available BEFORE the formation date.
        Must include the market ticker column.
    test_returns : pd.DataFrame
        Returns during the holding period AFTER the formation date.
        Same columns as formation_returns.
    market_ticker : str
        Which column represents the market benchmark.
    risk_free_rate : float
        Annualized risk-free rate as a decimal.
    asset_tickers : list
        Which columns are actual investable assets (excludes market_ticker).
    method : str
        One of "min_variance", "max_sharpe", "target_risk", "equal_weight".
    target_volatility : float, optional
        Required if method == "target_risk".

    Returns
    -------
    pd.Series
        Daily portfolio returns during the test period, using the
        weights determined from formation_returns only.
    """
    if method == "equal_weight":
        n = len(asset_tickers)
        weights = pd.Series([1 / n] * n, index=asset_tickers)

    else:
        capm_summary = build_capm_summary(formation_returns, market_ticker, risk_free_rate)
        capm_summary = calculate_capm_expected_returns(
            capm_summary, formation_returns[market_ticker], risk_free_rate
        )
        cov_matrix = calculate_covariance_matrix(formation_returns[asset_tickers])
        expected_returns = capm_summary["capm_expected_return"]

        if method == "min_variance":
            result = minimum_variance_portfolio(cov_matrix)
            weights = result["weights"]

        elif method == "max_sharpe":
            result = maximum_sharpe_portfolio(expected_returns, cov_matrix, risk_free_rate)
            weights = result["weights"]

        elif method == "target_risk":
            if target_volatility is None:
                raise ValueError("target_volatility is required for method='target_risk'")
            result = target_risk_portfolio(expected_returns, cov_matrix, target_volatility)
            weights = result["weights"]

        else:
            raise ValueError(f"Unknown method: {method}")

    # Apply the fixed formation-period weights to the test period's actual returns
    aligned_test_returns = test_returns[asset_tickers]
    portfolio_returns = aligned_test_returns.dot(weights)

    return portfolio_returns

def run_backtest(
    returns_df: pd.DataFrame,
    market_ticker: str,
    risk_free_rate: float,
    asset_tickers: list,
    method: str,
    target_volatility: float = None,
) -> pd.Series:
    """
    Run a full walk-forward backtest for one portfolio method across
    the entire dataset, stitching together each quarter's realized
    returns into one continuous series.

    Parameters
    ----------
    returns_df : pd.DataFrame
        Full daily returns history, all tickers, entire dataset.
    market_ticker : str
        Which column represents the market benchmark.
    risk_free_rate : float
        Annualized risk-free rate as a decimal.
    asset_tickers : list
        Which columns are actual investable assets.
    method : str
        One of "min_variance", "max_sharpe", "target_risk", "equal_weight".
    target_volatility : float, optional
        Required if method == "target_risk".

    Returns
    -------
    pd.Series
        Continuous daily portfolio returns across the entire
        backtest period (from the end of the first formation
        window through the end of the dataset).
    """
    rebalance_dates = generate_rebalance_dates(returns_df)
    all_period_returns = []

    for i, formation_date in enumerate(rebalance_dates):
        formation_returns = returns_df.loc[:formation_date].iloc[:-1]  # strictly before formation_date

        if i + 1 < len(rebalance_dates):
            next_date = rebalance_dates[i + 1]
            test_returns = returns_df.loc[formation_date:next_date].iloc[:-1]
        else:
            test_returns = returns_df.loc[formation_date:]

        if test_returns.empty:
            continue

        period_returns = run_single_period(
            formation_returns, test_returns, market_ticker,
            risk_free_rate, asset_tickers, method, target_volatility,
        )
        all_period_returns.append(period_returns)

    full_return_series = pd.concat(all_period_returns)
    return full_return_series


def calculate_cumulative_return(daily_returns: pd.Series) -> float:
    """Total compounded return over the full period."""
    return (1 + daily_returns).prod() - 1


def calculate_cagr(daily_returns: pd.Series, trading_days_per_year: int = 252) -> float:
    """Compound Annual Growth Rate."""
    total_return = calculate_cumulative_return(daily_returns)
    n_years = len(daily_returns) / trading_days_per_year
    return (1 + total_return) ** (1 / n_years) - 1


def calculate_max_drawdown(daily_returns: pd.Series) -> float:
    """
    Maximum peak-to-trough decline in cumulative portfolio value.
    Returned as a negative decimal (e.g. -0.23 for a 23% drawdown).
    """
    cumulative_value = (1 + daily_returns).cumprod()
    running_max = cumulative_value.cummax()
    drawdown = (cumulative_value - running_max) / running_max
    return drawdown.min()


def calculate_sortino_ratio(
    daily_returns: pd.Series,
    risk_free_rate: float,
    trading_days_per_year: int = 252,
) -> float:
    """
    Sortino ratio: like Sharpe, but only penalizes downside volatility.
    """
    daily_rf = risk_free_rate / trading_days_per_year
    excess_returns = daily_returns - daily_rf

    downside_returns = excess_returns[excess_returns < 0]
    downside_std = downside_returns.std()

    if downside_std == 0 or pd.isna(downside_std):
        return np.nan

    annualized_excess_return = excess_returns.mean() * trading_days_per_year
    annualized_downside_std = downside_std * np.sqrt(trading_days_per_year)

    return annualized_excess_return / annualized_downside_std


def summarize_backtest(
    daily_returns: pd.Series,
    risk_free_rate: float,
    betas: pd.Series = None,
    weights: pd.Series = None,
) -> dict:
    """
    Produce a full performance summary for a backtested return series.
    """
    from src.returns import annualize_return, annualize_volatility

    summary = {
        "cumulative_return": calculate_cumulative_return(daily_returns),
        "cagr": calculate_cagr(daily_returns),
        "annualized_volatility": annualize_volatility(daily_returns.std()),
        "sharpe_ratio": (annualize_return(daily_returns.mean()) - risk_free_rate) / annualize_volatility(daily_returns.std()),
        "max_drawdown": calculate_max_drawdown(daily_returns),
        "sortino_ratio": calculate_sortino_ratio(daily_returns, risk_free_rate),
    }

    return summary


if __name__ == "__main__":
    from src import database
    from src.returns import build_returns_dataframe
    from src.data_loader import fetch_risk_free_rate
    from config import MARKET_TICKER

    all_tickers = ["AAPL", "MSFT", "NVDA", "VTIP", "SPY"]
    price_data = {t: database.load_price_history(t) for t in all_tickers}
    returns_df = build_returns_dataframe(price_data)

    rf = fetch_risk_free_rate()
    asset_tickers = ["AAPL", "MSFT", "NVDA", "VTIP"]

    methods = {
        "Min Variance": ("min_variance", None),
        "Max Sharpe": ("max_sharpe", None),
        "Target Risk (15%)": ("target_risk", 0.15),
        "Equal Weight": ("equal_weight", None),
    }

    results = {}
    for name, (method, target_vol) in methods.items():
        print(f"Running backtest: {name}...")
        returns = run_backtest(returns_df, MARKET_TICKER, rf, asset_tickers, method, target_vol)
        results[name] = summarize_backtest(returns, rf)

    # Also backtest SPY itself as a pure buy-and-hold comparison
    spy_returns = returns_df[MARKET_TICKER]
    results["S&P 500 (SPY)"] = summarize_backtest(spy_returns, rf)

    results_df = pd.DataFrame(results).T
    print("\nBacktest Results:")
    print(results_df.round(4))
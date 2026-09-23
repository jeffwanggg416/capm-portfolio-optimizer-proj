import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252


def calculate_daily_returns(prices: pd.Series) -> pd.Series:
    """
    Calculate simple daily returns from a price series.
    R(t) = P(t) / P(t-1) - 1
    """
    returns = prices.pct_change()
    return returns.dropna()


def build_returns_dataframe(price_data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Combine multiple tickers' price histories into one aligned
    DataFrame of daily returns.

    Parameters
    ----------
    price_data : dict[str, pd.DataFrame]
        Maps ticker -> DataFrame with an 'adjusted_close' column,
        as returned by database.load_price_history().

    Returns
    -------
    pd.DataFrame
        Columns are tickers, index is date, values are daily
        simple returns. Rows with any missing values are dropped
        so all tickers are aligned on the same trading days.
    """
    return_series = {}

    for ticker, df in price_data.items():
        return_series[ticker] = calculate_daily_returns(df["adjusted_close"])

    returns_df = pd.DataFrame(return_series)
    returns_df = returns_df.dropna()

    return returns_df


def annualize_return(daily_mean_return: float) -> float:
    """Convert a mean daily return into an annualized return."""
    return daily_mean_return * TRADING_DAYS_PER_YEAR


def annualize_volatility(daily_std: float) -> float:
    """Convert daily volatility (std dev) into annualized volatility."""
    return daily_std * np.sqrt(TRADING_DAYS_PER_YEAR)


def summarize_returns(returns_df: pd.DataFrame) -> pd.DataFrame:
    """
    Produce a summary table of annualized return and volatility
    for each ticker in the returns DataFrame.
    """
    summary = pd.DataFrame({
        "annualized_return": returns_df.mean().apply(annualize_return),
        "annualized_volatility": returns_df.std().apply(annualize_volatility),
    })

    return summary
if __name__ == "__main__":
    from src import database

    tickers = ["AAPL", "MSFT", "NVDA", "VTIP"]
    price_data = {t: database.load_price_history(t) for t in tickers}

    # Note: this assumes you've already run data_loader for each of these
    # tickers at least once so they're cached in SQLite.

    returns_df = build_returns_dataframe(price_data)
    print(returns_df.head())
    print(f"\nShape: {returns_df.shape}")

    summary = summarize_returns(returns_df)
    print("\nSummary:")
    print(summary)
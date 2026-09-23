import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252


def calculate_covariance_matrix(returns_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the annualized covariance matrix from daily returns.

    Parameters
    ----------
    returns_df : pd.DataFrame
        Daily simple returns, columns are tickers.

    Returns
    -------
    pd.DataFrame
        Annualized covariance matrix, square, indexed and
        columned by ticker.
    """
    daily_cov = returns_df.cov()
    annualized_cov = daily_cov * TRADING_DAYS_PER_YEAR
    return annualized_cov


def calculate_correlation_matrix(returns_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the correlation matrix from daily returns.
    Correlation is scale-independent, so no annualization is needed.
    """
    return returns_df.corr()


def portfolio_variance(weights: np.ndarray, cov_matrix: pd.DataFrame) -> float:
    """
    Calculate portfolio variance: sigma_p^2 = w' Sigma w

    Parameters
    ----------
    weights : np.ndarray
        Portfolio weights, in the same order as cov_matrix's columns.
    cov_matrix : pd.DataFrame
        Annualized covariance matrix.

    Returns
    -------
    float
        Portfolio variance.
    """
    cov_values = cov_matrix.values
    return weights.T @ cov_values @ weights


def portfolio_volatility(weights: np.ndarray, cov_matrix: pd.DataFrame) -> float:
    """
    Calculate portfolio volatility: sigma_p = sqrt(w' Sigma w)
    """
    return np.sqrt(portfolio_variance(weights, cov_matrix))

if __name__ == "__main__":
    from src import database
    from src.returns import build_returns_dataframe

    tickers = ["AAPL", "MSFT", "NVDA", "VTIP", "SPY"]
    price_data = {t: database.load_price_history(t) for t in tickers}
    returns_df = build_returns_dataframe(price_data)

    cov_matrix = calculate_covariance_matrix(returns_df)
    corr_matrix = calculate_correlation_matrix(returns_df)

    print("Covariance Matrix (annualized):")
    print(cov_matrix)
    print("\nCorrelation Matrix:")
    print(corr_matrix)

    # Quick sanity test: equal-weighted portfolio across the 4 real assets (excluding SPY)
    import numpy as np
    equal_weights = np.array([0.25, 0.25, 0.25, 0.25])
    small_cov = cov_matrix.loc[["AAPL", "MSFT", "NVDA", "VTIP"], ["AAPL", "MSFT", "NVDA", "VTIP"]]

    vol = portfolio_volatility(equal_weights, small_cov)
    print(f"\nEqual-weighted (AAPL/MSFT/NVDA/VTIP) portfolio volatility: {vol:.4f}")    
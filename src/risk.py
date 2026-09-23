import numpy as np
import pandas as pd

from src.optimizer import target_risk_portfolio


def build_weight_shift_table(
    expected_returns: pd.Series,
    cov_matrix: pd.DataFrame,
    volatility_targets: list[float],
) -> pd.DataFrame:
    """
    Run the Target Risk optimizer across multiple volatility targets
    and assemble the resulting weights into one comparison table.

    Parameters
    ----------
    expected_returns : pd.Series
        CAPM expected returns, indexed by ticker.
    cov_matrix : pd.DataFrame
        Annualized covariance matrix.
    volatility_targets : list[float]
        Volatility targets as decimals, e.g. [0.08, 0.10, 0.12, ...]

    Returns
    -------
    pd.DataFrame
        Rows are tickers, columns are volatility targets (as
        percentages for readability, e.g. "8%"), values are weights.
        Infeasible targets are dropped with a printed warning.
    """
    weight_columns = {}

    for target in volatility_targets:
        result = target_risk_portfolio(expected_returns, cov_matrix, target)

        column_label = f"{target * 100:.0f}%"

        if not result["feasible"]:
            print(f"Warning: {column_label} target volatility is below the achievable minimum, skipping.")
            continue

        if not result["success"]:
            print(f"Warning: optimizer did not converge cleanly for {column_label} target.")

        weight_columns[column_label] = result["weights"]

    shift_table = pd.DataFrame(weight_columns)
    return shift_table


def calculate_weight_changes(shift_table: pd.DataFrame, from_column: str, to_column: str) -> pd.DataFrame:
    """
    Calculate the change in each asset's weight between two
    volatility target columns in a weight shift table.

    Parameters
    ----------
    shift_table : pd.DataFrame
        Output of build_weight_shift_table().
    from_column : str
        Column label of the starting target (e.g. "12%").
    to_column : str
        Column label of the ending target (e.g. "20%").

    Returns
    -------
    pd.DataFrame
        Original two columns plus a 'change' column.
    """
    comparison = shift_table[[from_column, to_column]].copy()
    comparison["change"] = comparison[to_column] - comparison[from_column]
    return comparison
def calculate_portfolio_beta(weights: pd.Series, betas: pd.Series) -> float:
    """
    Calculate portfolio beta as the weighted average of individual betas.

    Beta_p = sum(w_i * Beta_i)

    Parameters
    ----------
    weights : pd.Series
        Portfolio weights, indexed by ticker.
    betas : pd.Series
        Individual security betas, indexed by ticker.

    Returns
    -------
    float
        Portfolio beta.
    """
    aligned_betas = betas.loc[weights.index]
    return np.dot(weights.values, aligned_betas.values)


def summarize_portfolio(
    weights: pd.Series,
    expected_returns: pd.Series,
    betas: pd.Series,
    cov_matrix: pd.DataFrame,
    risk_free_rate: float,
) -> dict:
    """
    Produce a full risk/return summary for a given set of portfolio weights.

    Parameters
    ----------
    weights : pd.Series
        Portfolio weights, indexed by ticker.
    expected_returns : pd.Series
        CAPM expected returns, indexed by ticker.
    betas : pd.Series
        Individual security betas, indexed by ticker.
    cov_matrix : pd.DataFrame
        Annualized covariance matrix.
    risk_free_rate : float
        Annualized risk-free rate as a decimal.

    Returns
    -------
    dict with keys:
        expected_return, volatility, sharpe_ratio, beta
    """
    from src.covariance import portfolio_volatility

    aligned_returns = expected_returns.loc[weights.index]
    port_return = np.dot(weights.values, aligned_returns.values)
    port_vol = portfolio_volatility(weights.values, cov_matrix.loc[weights.index, weights.index])
    port_beta = calculate_portfolio_beta(weights, betas)
    sharpe = (port_return - risk_free_rate) / port_vol

    return {
        "expected_return": port_return,
        "volatility": port_vol,
        "sharpe_ratio": sharpe,
        "beta": port_beta,
    }



if __name__ == "__main__":
    from src import database
    from src.returns import build_returns_dataframe
    from src.covariance import calculate_covariance_matrix
    from src.capm import build_capm_summary, calculate_capm_expected_returns
    from src.data_loader import fetch_risk_free_rate
    from src.optimizer import minimum_variance_portfolio, maximum_sharpe_portfolio, target_risk_portfolio
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
    betas = capm_summary["beta"]

    gmv = minimum_variance_portfolio(cov_matrix)
    sharpe_port = maximum_sharpe_portfolio(expected_returns, cov_matrix, rf)
    target_port = target_risk_portfolio(expected_returns, cov_matrix, 0.15)

    for name, result in [("GMV", gmv), ("Max Sharpe", sharpe_port), ("Target Risk (15%)", target_port)]:
        summary = summarize_portfolio(result["weights"], expected_returns, betas, cov_matrix, rf)
        print(f"=== {name} ===")
        print(f"Expected return: {summary['expected_return']:.4f}")
        print(f"Volatility:      {summary['volatility']:.4f}")
        print(f"Sharpe ratio:    {summary['sharpe_ratio']:.4f}")
        print(f"Beta:            {summary['beta']:.4f}\n")
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from src.covariance import portfolio_variance, portfolio_volatility


def minimum_variance_portfolio(cov_matrix: pd.DataFrame) -> dict:
    """
    Solve for the Global Minimum Variance portfolio.

    Minimize:    w' Sigma w
    Subject to:  sum(w) = 1
                 0 <= w_i <= 1

    Parameters
    ----------
    cov_matrix : pd.DataFrame
        Annualized covariance matrix.

    Returns
    -------
    dict with keys:
        weights (pd.Series indexed by ticker),
        volatility (float),
        success (bool)
    """
    tickers = cov_matrix.columns.tolist()
    n_assets = len(tickers)

    def objective(weights):
        return portfolio_variance(weights, cov_matrix)

    constraints = [
        {"type": "eq", "fun": lambda w: np.sum(w) - 1}
    ]

    bounds = tuple((0, 1) for _ in range(n_assets))

    initial_guess = np.array([1 / n_assets] * n_assets)

    result = minimize(
        objective,
        initial_guess,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"ftol": 1e-9, "maxiter": 1000},
    )

    weights = pd.Series(result.x, index=tickers)

    return {
        "weights": weights,
        "volatility": portfolio_volatility(result.x, cov_matrix),
        "success": result.success,
    }

def maximum_sharpe_portfolio(
    expected_returns: pd.Series,
    cov_matrix: pd.DataFrame,
    risk_free_rate: float,
) -> dict:
    """
    Solve for the Maximum Sharpe Ratio portfolio.

    Maximize:    (E(Rp) - Rf) / sigma_p
    Subject to:  sum(w) = 1
                 0 <= w_i <= 1

    Implemented as minimizing the negative Sharpe ratio.

    Parameters
    ----------
    expected_returns : pd.Series
        CAPM expected returns, indexed by ticker, same order as cov_matrix.
    cov_matrix : pd.DataFrame
        Annualized covariance matrix.
    risk_free_rate : float
        Annualized risk-free rate as a decimal.

    Returns
    -------
    dict with keys:
        weights (pd.Series indexed by ticker),
        expected_return (float),
        volatility (float),
        sharpe_ratio (float),
        success (bool)
    """
    tickers = cov_matrix.columns.tolist()
    n_assets = len(tickers)
    exp_returns_array = expected_returns.loc[tickers].values

    def negative_sharpe(weights):
        port_return = np.dot(weights, exp_returns_array)
        port_vol = portfolio_volatility(weights, cov_matrix)
        sharpe = (port_return - risk_free_rate) / port_vol
        return -sharpe

    constraints = [
        {"type": "eq", "fun": lambda w: np.sum(w) - 1}
    ]

    bounds = tuple((0, 1) for _ in range(n_assets))

    initial_guess = np.array([1 / n_assets] * n_assets)

    result = minimize(
        negative_sharpe,
        initial_guess,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"ftol": 1e-9, "maxiter": 1000},
    )

    weights = pd.Series(result.x, index=tickers)
    port_return = np.dot(result.x, exp_returns_array)
    port_vol = portfolio_volatility(result.x, cov_matrix)
    sharpe = (port_return - risk_free_rate) / port_vol

    return {
        "weights": weights,
        "expected_return": port_return,
        "volatility": port_vol,
        "sharpe_ratio": sharpe,
        "success": result.success,
    }

def target_risk_portfolio(
    expected_returns: pd.Series,
    cov_matrix: pd.DataFrame,
    target_volatility: float,
) -> dict:
    """
    Solve for the portfolio that maximizes expected return subject
    to a volatility ceiling.

    Maximize:    E(Rp)
    Subject to:  sigma_p <= target_volatility
                 sum(w) = 1
                 0 <= w_i <= 1

    Implemented as minimizing the negative expected return.

    Parameters
    ----------
    expected_returns : pd.Series
        CAPM expected returns, indexed by ticker, same order as cov_matrix.
    cov_matrix : pd.DataFrame
        Annualized covariance matrix.
    target_volatility : float
        Maximum acceptable annualized volatility, as a decimal (e.g. 0.12 for 12%).

    Returns
    -------
    dict with keys:
        weights (pd.Series indexed by ticker),
        expected_return (float),
        volatility (float),
        success (bool),
        feasible (bool)
    """
    tickers = cov_matrix.columns.tolist()
    n_assets = len(tickers)
    exp_returns_array = expected_returns.loc[tickers].values

    # Feasibility check: target can't be below the achievable minimum
    gmv_result = minimum_variance_portfolio(cov_matrix)
    min_achievable_volatility = gmv_result["volatility"]

    if target_volatility < min_achievable_volatility:
        return {
            "weights": gmv_result["weights"],
            "expected_return": np.dot(gmv_result["weights"].values, exp_returns_array),
            "volatility": min_achievable_volatility,
            "success": False,
            "feasible": False,
        }

    def negative_expected_return(weights):
        return -np.dot(weights, exp_returns_array)

    constraints = [
        {"type": "eq", "fun": lambda w: np.sum(w) - 1},
        {"type": "ineq", "fun": lambda w: target_volatility - portfolio_volatility(w, cov_matrix)},
    ]

    bounds = tuple((0, 1) for _ in range(n_assets))

    initial_guess = np.array([1 / n_assets] * n_assets)

    result = minimize(
        negative_expected_return,
        initial_guess,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"ftol": 1e-9, "maxiter": 1000},
    )

    weights = pd.Series(result.x, index=tickers)
    port_return = np.dot(result.x, exp_returns_array)
    port_vol = portfolio_volatility(result.x, cov_matrix)

    return {
        "weights": weights,
        "expected_return": port_return,
        "volatility": port_vol,
        "success": result.success,
        "feasible": True,
    }

import numpy as np
import pandas as pd
import pytest

from src.optimizer import minimum_variance_portfolio, maximum_sharpe_portfolio, target_risk_portfolio
from src.covariance import portfolio_volatility


@pytest.fixture
def sample_cov_matrix():
    """A small, hand-constructed covariance matrix for testing."""
    tickers = ["A", "B", "C"]
    data = np.array([
        [0.04, 0.01, 0.00],
        [0.01, 0.09, 0.02],
        [0.00, 0.02, 0.01],
    ])
    return pd.DataFrame(data, index=tickers, columns=tickers)


@pytest.fixture
def sample_expected_returns():
    """Expected returns matching sample_cov_matrix's tickers."""
    return pd.Series({"A": 0.08, "B": 0.15, "C": 0.05})


def test_gmv_weights_sum_to_one(sample_cov_matrix):
    result = minimum_variance_portfolio(sample_cov_matrix)
    assert abs(result["weights"].sum() - 1) < 1e-6


def test_gmv_weights_non_negative(sample_cov_matrix):
    result = minimum_variance_portfolio(sample_cov_matrix)
    assert (result["weights"] >= -1e-8).all()  # tiny numerical tolerance


def test_gmv_weights_not_above_one(sample_cov_matrix):
    result = minimum_variance_portfolio(sample_cov_matrix)
    assert (result["weights"] <= 1 + 1e-8).all()


def test_gmv_beats_random_portfolios(sample_cov_matrix):
    """
    The GMV portfolio should have volatility less than or equal to
    a large sample of random long-only portfolios.
    """
    gmv_result = minimum_variance_portfolio(sample_cov_matrix)
    gmv_vol = gmv_result["volatility"]

    n_assets = len(sample_cov_matrix)
    worse_count = 0

    np.random.seed(42)
    for _ in range(1000):
        raw = np.random.random(n_assets)
        random_weights = raw / raw.sum()
        random_vol = portfolio_volatility(random_weights, sample_cov_matrix)
        if random_vol >= gmv_vol:
            worse_count += 1

    # GMV should beat (or tie) essentially all random portfolios
    assert worse_count >= 990  # allow a tiny numerical tolerance margin


def test_max_sharpe_beats_gmv_sharpe(sample_cov_matrix, sample_expected_returns):
    """Max Sharpe portfolio should have a Sharpe ratio >= GMV's Sharpe ratio."""
    rf = 0.03

    gmv_result = minimum_variance_portfolio(sample_cov_matrix)
    sharpe_result = maximum_sharpe_portfolio(sample_expected_returns, sample_cov_matrix, rf)

    gmv_return = np.dot(gmv_result["weights"].values, sample_expected_returns.values)
    gmv_sharpe = (gmv_return - rf) / gmv_result["volatility"]

    assert sharpe_result["sharpe_ratio"] >= gmv_sharpe - 1e-6


def test_max_sharpe_weights_sum_to_one(sample_cov_matrix, sample_expected_returns):
    result = maximum_sharpe_portfolio(sample_expected_returns, sample_cov_matrix, 0.03)
    assert abs(result["weights"].sum() - 1) < 1e-6


def test_target_risk_respects_constraint(sample_cov_matrix, sample_expected_returns):
    """Target Risk portfolio's realized volatility should not exceed the target
    by more than a small numerical tolerance."""
    target = 0.15
    result = target_risk_portfolio(sample_expected_returns, sample_cov_matrix, target)

    assert result["feasible"]
    assert result["volatility"] <= target + 1e-4


def test_target_risk_infeasible_below_gmv(sample_cov_matrix, sample_expected_returns):
    """A target volatility below GMV's achievable minimum should be flagged infeasible."""
    gmv_result = minimum_variance_portfolio(sample_cov_matrix)
    impossible_target = gmv_result["volatility"] * 0.5

    result = target_risk_portfolio(sample_expected_returns, sample_cov_matrix, impossible_target)
    assert result["feasible"] is False


def test_covariance_matrix_dimensions(sample_cov_matrix):
    """Covariance matrix should be square, with matching row/column labels."""
    assert sample_cov_matrix.shape[0] == sample_cov_matrix.shape[1]
    assert list(sample_cov_matrix.index) == list(sample_cov_matrix.columns)
import pandas as pd
from qdlib.empirical.log_returns import compute_log_returns, annualized_mean_vol


def test_compute_log_returns_length():
    prices = pd.Series([100, 101, 102, 100])
    returns = compute_log_returns(prices)
    assert len(returns) == 3


def test_annualized_mean_vol_positive_vol():
    prices = pd.Series([100, 101, 99, 103, 102, 105])
    returns = compute_log_returns(prices)
    mu, sigma = annualized_mean_vol(returns)
    assert sigma > 0

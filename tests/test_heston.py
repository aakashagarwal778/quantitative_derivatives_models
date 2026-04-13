import numpy as np
from qdlib.stochastic_volatility.heston_fft import heston_price_grid
from qdlib.stochastic_volatility.heston_implied_volatility import heston_implied_volatility


def test_heston_grid_returns_positive_prices():
    strikes = np.array([80, 90, 100, 110], dtype=float)
    prices = heston_price_grid(strikes, 100, 1.0, 0.03, 1.5, 0.04, 0.4, -0.6, 0.04)
    assert np.all(prices > 0)


def test_heston_prices_decrease_with_strike():
    strikes = np.array([80, 90, 100, 110], dtype=float)
    prices = heston_price_grid(strikes, 100, 1.0, 0.03, 1.5, 0.04, 0.4, -0.6, 0.04)
    assert np.all(np.diff(prices) < 0)


def test_heston_implied_vol_positive():
    iv = heston_implied_volatility(100, 100, 1.0, 0.03, 1.5, 0.04, 0.4, -0.6, 0.04)
    assert iv > 0

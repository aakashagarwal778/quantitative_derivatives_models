import numpy as np
from qdlib.calibration.heston_calibration import calibrate_heston


def test_heston_calibration_returns_feasible_params():
    strikes = np.array([90, 100, 110], dtype=float)
    maturities = np.array([0.5, 1.0, 1.5], dtype=float)
    rates = np.full(3, 0.03)
    market_prices = np.array([14.0, 9.5, 6.5], dtype=float)
    res = calibrate_heston(100, strikes, maturities, rates, market_prices)
    assert res.success
    kappa, theta, sigma, rho, v0 = res.x
    assert kappa > 0 and theta > 0 and sigma > 0 and v0 > 0 and -1 < rho < 1

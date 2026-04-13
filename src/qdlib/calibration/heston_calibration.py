from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from ..stochastic_volatility.heston_fft import heston_price_grid


def calibrate_heston(
    spot: float,
    strikes: np.ndarray,
    maturities: np.ndarray,
    rates: np.ndarray,
    market_prices: np.ndarray,
    initial_guess=(1.5, 0.04, 0.4, -0.5, 0.04),
):
    """Calibrate Heston parameters by least squares on option prices."""
    strikes = np.asarray(strikes, dtype=float)
    maturities = np.asarray(maturities, dtype=float)
    rates = np.asarray(rates, dtype=float)
    market_prices = np.asarray(market_prices, dtype=float)

    def objective(params):
        kappa, theta, sigma, rho, v0 = params
        if kappa <= 0 or theta <= 0 or sigma <= 0 or v0 <= 0 or not -0.999 < rho < 0.999:
            return 1e9
        model_prices = np.array([
            heston_price_grid([k], spot, t, r, kappa, theta, sigma, rho, v0)[0]
            for k, t, r in zip(strikes, maturities, rates)
        ])
        scale = np.maximum(market_prices, 1.0)
        return float(np.mean(((model_prices - market_prices) / scale) ** 2))

    bounds = [(1e-4, 10.0), (1e-5, 2.0), (1e-4, 5.0), (-0.999, 0.999), (1e-5, 2.0)]
    return minimize(objective, np.array(initial_guess, dtype=float), method='L-BFGS-B', bounds=bounds)

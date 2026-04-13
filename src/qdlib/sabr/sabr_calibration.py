from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from .sabr_implied_volatility import sabr_black_implied_vol


def calibrate_sabr(strikes: np.ndarray, market_vols: np.ndarray, forward: float, maturity: float, beta: float, initial_guess=(0.2, -0.2, 0.5)):
    strikes = np.asarray(strikes, dtype=float)
    market_vols = np.asarray(market_vols, dtype=float)

    def objective(params):
        alpha, rho, nu = params
        if alpha <= 0 or nu <= 0 or not -0.999 < rho < 0.999:
            return 1e9
        model = np.array([sabr_black_implied_vol(forward, k, maturity, alpha, beta, rho, nu) for k in strikes])
        return float(np.mean((model - market_vols) ** 2))

    bounds = [(1e-6, 5.0), (-0.999, 0.999), (1e-6, 5.0)]
    result = minimize(objective, np.array(initial_guess, dtype=float), method='L-BFGS-B', bounds=bounds)
    return result

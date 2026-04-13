from __future__ import annotations

import numpy as np
from .heston_pricing import heston_call_price


def heston_price_grid(strikes, spot: float, maturity: float, rate: float, kappa: float, theta: float, sigma: float, rho: float, v0: float) -> np.ndarray:
    strikes = np.asarray(strikes, dtype=float)
    return np.array([heston_call_price(spot, k, maturity, rate, kappa, theta, sigma, rho, v0) for k in strikes])

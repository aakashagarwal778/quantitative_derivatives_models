from __future__ import annotations

import math
import cmath
from scipy.integrate import quad
from .heston_characteristic import heston_characteristic_function


def _probability(j: int, spot: float, strike: float, maturity: float, rate: float, kappa: float, theta: float, sigma: float, rho: float, v0: float) -> float:
    ln_k = math.log(strike)
    if j == 1:
        shift = -1j
        denom_cf = heston_characteristic_function(-1j, spot, maturity, rate, kappa, theta, sigma, rho, v0)
        def integrand(u: float) -> float:
            val = heston_characteristic_function(u + shift, spot, maturity, rate, kappa, theta, sigma, rho, v0)
            return (cmath.exp(-1j * u * ln_k) * val / (1j * u * denom_cf)).real
    else:
        def integrand(u: float) -> float:
            val = heston_characteristic_function(u, spot, maturity, rate, kappa, theta, sigma, rho, v0)
            return (cmath.exp(-1j * u * ln_k) * val / (1j * u)).real
    integral, _ = quad(integrand, 1e-9, 100.0, limit=250)
    return 0.5 + integral / math.pi


def heston_call_price(spot: float, strike: float, maturity: float, rate: float, kappa: float, theta: float, sigma: float, rho: float, v0: float) -> float:
    p1 = _probability(1, spot, strike, maturity, rate, kappa, theta, sigma, rho, v0)
    p2 = _probability(2, spot, strike, maturity, rate, kappa, theta, sigma, rho, v0)
    return spot * p1 - strike * math.exp(-rate * maturity) * p2

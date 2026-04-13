from __future__ import annotations

import cmath


def heston_characteristic_function(u: complex, spot: float, maturity: float, rate: float, kappa: float, theta: float, sigma: float, rho: float, v0: float) -> complex:
    x0 = cmath.log(spot)
    a = kappa * theta
    b = kappa
    d = cmath.sqrt((rho * sigma * 1j * u - b) ** 2 + sigma**2 * (1j * u + u**2))
    g = (b - rho * sigma * 1j * u - d) / (b - rho * sigma * 1j * u + d)
    exp_dt = cmath.exp(-d * maturity)
    C = (rate * 1j * u * maturity + a / sigma**2 * ((b - rho * sigma * 1j * u - d) * maturity - 2 * cmath.log((1 - g * exp_dt) / (1 - g))))
    D = ((b - rho * sigma * 1j * u - d) / sigma**2) * ((1 - exp_dt) / (1 - g * exp_dt))
    return cmath.exp(C + D * v0 + 1j * u * x0)

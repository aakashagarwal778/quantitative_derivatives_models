from __future__ import annotations

import cmath
import math
from scipy.integrate import quad


def laplace_call_price_black_scholes(spot: float, strike: float, maturity: float, rate: float, volatility: float, alpha: float = 1.5) -> float:
    k = math.log(strike)

    def char_fn(u: complex) -> complex:
        mu = math.log(spot) + (rate - 0.5 * volatility**2) * maturity
        return cmath.exp(1j * u * mu - 0.5 * volatility**2 * maturity * u * u)

    def integrand(u: float) -> float:
        z = u - 1j * (alpha + 1.0)
        numerator = cmath.exp(-1j * u * k) * math.exp(-rate * maturity) * char_fn(z)
        denominator = alpha**2 + alpha - u*u + 1j * (2 * alpha + 1) * u
        return (numerator / denominator).real

    integral, _ = quad(integrand, 0.0, 150.0, limit=300)
    return math.exp(-alpha * k) * integral / math.pi

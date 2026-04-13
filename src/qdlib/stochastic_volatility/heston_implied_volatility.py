from __future__ import annotations

from ..pricing_foundations.implied_volatility import implied_volatility_from_price
from .heston_pricing import heston_call_price


def heston_implied_volatility(spot: float, strike: float, maturity: float, rate: float, kappa: float, theta: float, sigma: float, rho: float, v0: float) -> float:
    price = heston_call_price(spot, strike, maturity, rate, kappa, theta, sigma, rho, v0)
    return implied_volatility_from_price(price, spot, strike, maturity, rate, 'call')

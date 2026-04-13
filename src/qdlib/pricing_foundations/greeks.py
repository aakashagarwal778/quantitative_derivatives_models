from __future__ import annotations

import math
from ..common.utils import norm_cdf, norm_pdf
from ..pricing_foundations.black_scholes import d1_d2


def black_scholes_greeks(spot: float, strike: float, maturity: float, rate: float, volatility: float, option_type: str = 'call', dividend: float = 0.0) -> dict[str, float]:
    d1, d2 = d1_d2(spot, strike, maturity, rate, volatility, dividend)
    disc_q = math.exp(-dividend * maturity)
    disc_r = math.exp(-rate * maturity)
    sqrt_t = math.sqrt(maturity)

    if option_type.lower() == 'call':
        delta = disc_q * norm_cdf(d1)
        theta = (-spot * disc_q * norm_pdf(d1) * volatility / (2 * sqrt_t)
                 - rate * strike * disc_r * norm_cdf(d2)
                 + dividend * spot * disc_q * norm_cdf(d1))
        rho = strike * maturity * disc_r * norm_cdf(d2)
    elif option_type.lower() == 'put':
        delta = disc_q * (norm_cdf(d1) - 1)
        theta = (-spot * disc_q * norm_pdf(d1) * volatility / (2 * sqrt_t)
                 + rate * strike * disc_r * norm_cdf(-d2)
                 - dividend * spot * disc_q * norm_cdf(-d1))
        rho = -strike * maturity * disc_r * norm_cdf(-d2)
    else:
        raise ValueError("option_type must be 'call' or 'put'.")

    gamma = disc_q * norm_pdf(d1) / (spot * volatility * sqrt_t)
    vega = spot * disc_q * norm_pdf(d1) * sqrt_t
    return {"delta": delta, "gamma": gamma, "vega": vega, "theta": theta, "rho": rho}

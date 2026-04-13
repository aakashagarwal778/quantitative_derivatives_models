from __future__ import annotations

import math
from ..common.utils import norm_cdf, validate_positive


def d1_d2(spot: float, strike: float, maturity: float, rate: float, volatility: float, dividend: float = 0.0) -> tuple[float, float]:
    validate_positive(spot=spot, strike=strike, maturity=maturity, volatility=volatility)
    vt = volatility * math.sqrt(maturity)
    d1 = (math.log(spot / strike) + (rate - dividend + 0.5 * volatility * volatility) * maturity) / vt
    d2 = d1 - vt
    return d1, d2


def black_scholes_price(spot: float, strike: float, maturity: float, rate: float, volatility: float, option_type: str = 'call', dividend: float = 0.0) -> float:
    d1, d2 = d1_d2(spot, strike, maturity, rate, volatility, dividend)
    disc_q = math.exp(-dividend * maturity)
    disc_r = math.exp(-rate * maturity)
    option_type = option_type.lower()
    if option_type == 'call':
        return spot * disc_q * norm_cdf(d1) - strike * disc_r * norm_cdf(d2)
    if option_type == 'put':
        return strike * disc_r * norm_cdf(-d2) - spot * disc_q * norm_cdf(-d1)
    raise ValueError("option_type must be 'call' or 'put'.")

from __future__ import annotations

import math


def perpetual_american_put_price(spot: float, strike: float, rate: float, volatility: float) -> float:
    if rate <= 0:
        raise ValueError("rate must be positive for perpetual put formula")
    sigma2 = volatility * volatility
    beta = (0.5 - rate / sigma2) - math.sqrt((rate / sigma2 - 0.5)**2 + 2 * rate / sigma2)
    s_star = strike * beta / (beta - 1)
    if spot <= s_star:
        return strike - spot
    return (strike - s_star) * (spot / s_star) ** beta

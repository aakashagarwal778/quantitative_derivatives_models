from __future__ import annotations

import math
import numpy as np
from ..common.utils import validate_positive


def crr_parameters(rate: float, volatility: float, maturity: float, steps: int) -> tuple[float, float, float, float]:
    validate_positive(volatility=volatility, maturity=maturity)
    if steps <= 0:
        raise ValueError("steps must be positive")
    dt = maturity / steps
    u = math.exp(volatility * math.sqrt(dt))
    d = 1.0 / u
    p = (math.exp(rate * dt) - d) / (u - d)
    if not 0 <= p <= 1:
        raise ValueError("CRR risk-neutral probability fell outside [0, 1].")
    return dt, u, d, p


def binomial_option_price(spot: float, strike: float, maturity: float, rate: float, volatility: float, steps: int, option_type: str = 'call', american: bool = False) -> float:
    validate_positive(spot=spot, strike=strike, maturity=maturity, volatility=volatility)
    dt, u, d, p = crr_parameters(rate, volatility, maturity, steps)
    discount = math.exp(-rate * dt)
    j = np.arange(steps + 1)
    stock = spot * (u ** j) * (d ** (steps - j))
    if option_type.lower() == 'call':
        values = np.maximum(stock - strike, 0.0)
    elif option_type.lower() == 'put':
        values = np.maximum(strike - stock, 0.0)
    else:
        raise ValueError("option_type must be 'call' or 'put'.")

    for n in range(steps - 1, -1, -1):
        values = discount * (p * values[1:] + (1 - p) * values[:-1])
        if american:
            j = np.arange(n + 1)
            stock = spot * (u ** j) * (d ** (n - j))
            if option_type.lower() == 'call':
                exercise = np.maximum(stock - strike, 0.0)
            else:
                exercise = np.maximum(strike - stock, 0.0)
            values = np.maximum(values, exercise)
    return float(values[0])

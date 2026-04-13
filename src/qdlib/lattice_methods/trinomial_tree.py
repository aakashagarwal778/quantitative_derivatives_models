from __future__ import annotations

import math
import numpy as np
from ..common.utils import validate_positive


def trinomial_option_price(spot: float, strike: float, maturity: float, rate: float, volatility: float, steps: int, option_type: str = 'call', american: bool = False) -> float:
    validate_positive(spot=spot, strike=strike, maturity=maturity, volatility=volatility)
    if steps <= 0:
        raise ValueError("steps must be positive")
    dt = maturity / steps
    dx = volatility * math.sqrt(3 * dt)
    u = math.exp(dx)
    d = math.exp(-dx)
    nu = rate - 0.5 * volatility * volatility
    pu = 0.5 * ((volatility * volatility * dt + nu * nu * dt * dt) / (dx * dx) + (nu * dt) / dx)
    pd = 0.5 * ((volatility * volatility * dt + nu * nu * dt * dt) / (dx * dx) - (nu * dt) / dx)
    pm = 1.0 - pu - pd
    if min(pu, pm, pd) < -1e-12:
        raise ValueError("trinomial probabilities are invalid for this parameter set")
    pu, pm, pd = max(pu, 0.0), max(pm, 0.0), max(pd, 0.0)
    discount = math.exp(-rate * dt)

    idx = np.arange(-steps, steps + 1)
    stock = spot * np.exp(idx * dx)
    if option_type.lower() == 'call':
        values = np.maximum(stock - strike, 0.0)
    else:
        values = np.maximum(strike - stock, 0.0)

    for n in range(steps - 1, -1, -1):
        new_vals = np.empty(2 * n + 1)
        for i in range(2 * n + 1):
            new_vals[i] = discount * (pu * values[i + 2] + pm * values[i + 1] + pd * values[i])
        values = new_vals
        if american:
            idx = np.arange(-n, n + 1)
            stock = spot * np.exp(idx * dx)
            exercise = np.maximum(stock - strike, 0.0) if option_type.lower() == 'call' else np.maximum(strike - stock, 0.0)
            values = np.maximum(values, exercise)
    return float(values[0])

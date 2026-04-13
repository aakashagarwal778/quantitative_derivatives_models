from __future__ import annotations

import math
import numpy as np


def explicit_fd_european_put(spot: float, strike: float, maturity: float, rate: float, volatility: float, s_max: float | None = None, n_space: int = 200, n_time: int = 0) -> float:
    s_max = s_max or 4 * strike
    ds = s_max / n_space
    if n_time <= 0:
        # stability-oriented choice for the explicit scheme
        n_time = max(2000, int(1.2 * volatility**2 * n_space**2 * maturity) + 1)
    dt = maturity / n_time
    s = np.linspace(0, s_max, n_space + 1)
    values = np.maximum(strike - s, 0.0)
    j = np.arange(1, n_space)
    alpha = 0.5 * dt * (volatility**2 * j**2 - rate * j)
    beta = 1 - dt * (volatility**2 * j**2 + rate)
    gamma = 0.5 * dt * (volatility**2 * j**2 + rate * j)
    for n in range(n_time - 1, -1, -1):
        t = n * dt
        left = strike * math.exp(-rate * (maturity - t))
        next_values = values.copy()
        next_values[1:-1] = alpha * values[:-2] + beta * values[1:-1] + gamma * values[2:]
        next_values[0] = left
        next_values[-1] = 0.0
        values = next_values
    return float(np.interp(spot, s, values))

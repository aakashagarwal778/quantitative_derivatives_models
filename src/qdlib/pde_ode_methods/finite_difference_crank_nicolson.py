from __future__ import annotations

import numpy as np
from scipy.linalg import solve_banded


def crank_nicolson_european_put(spot: float, strike: float, maturity: float, rate: float, volatility: float, s_max: float | None = None, n_space: int = 200, n_time: int = 400) -> float:
    s_max = s_max or 4 * strike
    ds = s_max / n_space
    dt = maturity / n_time
    s = np.linspace(0, s_max, n_space + 1)
    values = np.maximum(strike - s, 0.0)
    j = np.arange(1, n_space)
    a = 0.25 * dt * (volatility**2 * j**2 - rate * j)
    b = -0.5 * dt * (volatility**2 * j**2 + rate)
    c = 0.25 * dt * (volatility**2 * j**2 + rate * j)
    ab = np.zeros((3, n_space - 1))
    ab[0, 1:] = -c[:-1]
    ab[1, :] = 1 - b
    ab[2, :-1] = -a[1:]
    for n in range(n_time - 1, -1, -1):
        tau = n * dt
        left_now = strike * np.exp(-rate * (maturity - tau))
        left_next = strike * np.exp(-rate * (maturity - (tau + dt)))
        rhs = a * values[:-2] + (1 + b) * values[1:-1] + c * values[2:]
        rhs[0] += a[0] * (left_now + left_next)
        values[1:-1] = solve_banded((1, 1), ab, rhs)
        values[0] = left_now
        values[-1] = 0.0
    return float(np.interp(spot, s, values))

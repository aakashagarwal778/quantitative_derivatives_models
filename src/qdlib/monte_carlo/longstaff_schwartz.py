from __future__ import annotations

import math
import numpy as np
from ..monte_carlo.gbm_simulation import simulate_gbm_paths


def longstaff_schwartz_american_put(spot: float, strike: float, maturity: float, rate: float, volatility: float, n_paths: int, n_steps: int, dividend: float = 0.0, seed: int | None = None) -> float:
    paths = simulate_gbm_paths(spot, maturity, rate, volatility, n_paths, n_steps, dividend, seed)
    dt = maturity / n_steps
    discount = math.exp(-rate * dt)
    cashflows = np.maximum(strike - paths[:, -1], 0.0)

    for t in range(n_steps - 1, 0, -1):
        cashflows *= discount
        intrinsic = np.maximum(strike - paths[:, t], 0.0)
        itm = intrinsic > 0
        if not np.any(itm):
            continue
        x = paths[itm, t]
        y = cashflows[itm]
        basis = np.column_stack([np.ones_like(x), x, x * x])
        coeffs, *_ = np.linalg.lstsq(basis, y, rcond=None)
        continuation = basis @ coeffs
        exercise = intrinsic[itm] > continuation
        idx = np.where(itm)[0]
        ex_idx = idx[exercise]
        cashflows[ex_idx] = intrinsic[ex_idx]
    return float(np.mean(cashflows) * discount)

from __future__ import annotations

from scipy.optimize import brentq


def solve_scalar_root(func, lower: float, upper: float, *, xtol: float = 1e-10) -> float:
    return float(brentq(func, lower, upper, xtol=xtol))

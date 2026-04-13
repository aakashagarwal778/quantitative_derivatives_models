from __future__ import annotations

import numpy as np


def simulate_gbm_terminal(spot: float, maturity: float, rate: float, volatility: float, n_paths: int, dividend: float = 0.0, seed: int | None = None, antithetic: bool = False) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if antithetic:
        n_half = (n_paths + 1) // 2
        z = rng.standard_normal(n_half)
        z = np.concatenate([z, -z])[:n_paths]
    else:
        z = rng.standard_normal(n_paths)
    drift = (rate - dividend - 0.5 * volatility**2) * maturity
    diffusion = volatility * np.sqrt(maturity) * z
    return spot * np.exp(drift + diffusion)


def simulate_gbm_paths(spot: float, maturity: float, rate: float, volatility: float, n_paths: int, n_steps: int, dividend: float = 0.0, seed: int | None = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    dt = maturity / n_steps
    paths = np.empty((n_paths, n_steps + 1), dtype=float)
    paths[:, 0] = spot
    shocks = rng.standard_normal((n_paths, n_steps))
    drift = (rate - dividend - 0.5 * volatility**2) * dt
    vol_dt = volatility * np.sqrt(dt)
    for t in range(n_steps):
        paths[:, t + 1] = paths[:, t] * np.exp(drift + vol_dt * shocks[:, t])
    return paths

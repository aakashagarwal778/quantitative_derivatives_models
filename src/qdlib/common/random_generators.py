from __future__ import annotations

import numpy as np


def standard_normals(n_paths: int, n_steps: int | None = None, seed: int | None = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if n_steps is None:
        return rng.standard_normal(n_paths)
    return rng.standard_normal((n_paths, n_steps))

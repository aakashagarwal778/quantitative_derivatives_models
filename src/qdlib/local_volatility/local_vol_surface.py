from __future__ import annotations

import numpy as np
from scipy.interpolate import RectBivariateSpline


class LocalVolSurface:
    def __init__(self, maturities: np.ndarray, strikes: np.ndarray, local_vols: np.ndarray) -> None:
        self.maturities = np.asarray(maturities, dtype=float)
        self.strikes = np.asarray(strikes, dtype=float)
        self.local_vols = np.asarray(local_vols, dtype=float)
        kx = min(3, len(self.maturities) - 1)
        ky = min(3, len(self.strikes) - 1)
        self._interp = RectBivariateSpline(self.maturities, self.strikes, self.local_vols, kx=kx, ky=ky)

    def __call__(self, maturity: float, strike: float) -> float:
        return float(self._interp(maturity, strike)[0, 0])

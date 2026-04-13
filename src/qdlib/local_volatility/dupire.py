from __future__ import annotations

import numpy as np
from ..pricing_foundations.black_scholes import black_scholes_price


def black_scholes_call_surface(spot: float, strikes: np.ndarray, maturities: np.ndarray, rate: float, volatility: float, dividend: float = 0.0) -> np.ndarray:
    strikes = np.asarray(strikes, dtype=float)
    maturities = np.asarray(maturities, dtype=float)
    surface = np.empty((len(maturities), len(strikes)), dtype=float)
    for i, t in enumerate(maturities):
        for j, k in enumerate(strikes):
            surface[i, j] = black_scholes_price(spot, float(k), float(t), rate, volatility, 'call', dividend)
    return surface


def dupire_local_variance(call_prices: np.ndarray, strikes: np.ndarray, maturities: np.ndarray, rate: float, dividend: float = 0.0) -> np.ndarray:
    """Estimate local variance from a call surface using Dupire's formula.

    The implementation assumes prices are observed on a regular grid in maturity and strike
    and uses finite differences. It is intended for educational use on smooth surfaces.
    """
    call_prices = np.asarray(call_prices, dtype=float)
    strikes = np.asarray(strikes, dtype=float)
    maturities = np.asarray(maturities, dtype=float)
    if call_prices.shape != (len(maturities), len(strikes)):
        raise ValueError('call_prices must have shape (len(maturities), len(strikes))')
    dC_dT = np.gradient(call_prices, maturities, axis=0, edge_order=2)
    dC_dK = np.gradient(call_prices, strikes, axis=1, edge_order=2)
    d2C_dK2 = np.gradient(dC_dK, strikes, axis=1, edge_order=2)
    kk = strikes[None, :]
    numerator = dC_dT + (rate - dividend) * kk * dC_dK + dividend * call_prices
    denominator = 0.5 * kk**2 * d2C_dK2
    with np.errstate(divide='ignore', invalid='ignore'):
        lv2 = numerator / denominator
    return np.maximum(lv2, 0.0)

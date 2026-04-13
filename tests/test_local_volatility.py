import numpy as np
from qdlib.local_volatility.dupire import black_scholes_call_surface, dupire_local_variance
from qdlib.local_volatility.local_vol_surface import LocalVolSurface


def test_dupire_on_black_scholes_surface_recovers_near_constant_vol():
    strikes = np.linspace(80, 120, 41)
    maturities = np.linspace(0.25, 2.0, 25)
    sigma = 0.2
    prices = black_scholes_call_surface(100, strikes, maturities, 0.03, sigma)
    lv2 = dupire_local_variance(prices, strikes, maturities, 0.03)
    interior = np.sqrt(lv2[3:-3, 3:-3])
    assert abs(interior.mean() - sigma) < 0.03


def test_local_vol_surface_interpolates_positive_values():
    maturities = np.array([0.5, 1.0, 2.0])
    strikes = np.array([90, 100, 110])
    vols = np.array([[0.22,0.20,0.21],[0.23,0.21,0.22],[0.24,0.22,0.23]])
    surf = LocalVolSurface(maturities, strikes, vols)
    assert surf(1.2, 105) > 0

import numpy as np
from qdlib.sabr.sabr_implied_volatility import sabr_black_implied_vol
from qdlib.sabr.sabr_calibration import calibrate_sabr


def test_sabr_vol_positive():
    assert sabr_black_implied_vol(100, 100, 1.0, 0.2, 0.5, -0.2, 0.6) > 0


def test_sabr_calibration_recovers_synthetic_smile():
    strikes = np.array([80,90,100,110,120], dtype=float)
    vols = np.array([sabr_black_implied_vol(100,k,1.0,0.2,0.5,-0.2,0.6) for k in strikes])
    res = calibrate_sabr(strikes, vols, 100, 1.0, 0.5, initial_guess=(0.18,-0.1,0.5))
    assert res.success
    model = np.array([sabr_black_implied_vol(100,k,1.0,*res.x[:1],0.5,res.x[1],res.x[2]) for k in strikes])
    assert np.mean((model - vols)**2) < 5e-7

from qdlib.jump_models.merton_jump_diffusion import merton_jump_diffusion_call
from qdlib.pricing_foundations.black_scholes import black_scholes_price


def test_merton_with_zero_jump_intensity_reduces_to_black_scholes():
    mjd = merton_jump_diffusion_call(100, 100, 1.0, 0.03, 0.2, 0.0, -0.1, 0.25)
    bs = black_scholes_price(100, 100, 1.0, 0.03, 0.2, 'call')
    assert abs(mjd - bs) < 1e-10


def test_merton_call_positive():
    mjd = merton_jump_diffusion_call(100, 100, 1.0, 0.03, 0.2, 0.4, -0.1, 0.25)
    assert mjd > 0

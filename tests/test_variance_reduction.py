from qdlib.monte_carlo.european_monte_carlo import european_option_price_mc
from qdlib.monte_carlo.variance_reduction import european_option_price_antithetic, importance_sampling_deep_otm_call
from qdlib.pricing_foundations.black_scholes import black_scholes_price


def test_antithetic_returns_valid_price():
    res = european_option_price_antithetic(100, 100, 1.0, 0.05, 0.2, 100000, 'call', seed=42)
    assert res.price > 0
    assert res.stderr is not None


def test_importance_sampling_deep_otm_close_to_black_scholes():
    res = importance_sampling_deep_otm_call(100, 140, 1.0, 0.05, 0.2, 150000, shift=1.6, seed=11)
    bs = black_scholes_price(100, 140, 1.0, 0.05, 0.2, 'call')
    assert abs(res.price - bs) < 0.15

from qdlib.monte_carlo.european_monte_carlo import european_option_price_mc
from qdlib.pricing_foundations.black_scholes import black_scholes_price


def test_monte_carlo_call_close_to_black_scholes():
    res = european_option_price_mc(100, 100, 1.0, 0.05, 0.2, 200000, 'call', seed=7)
    bs = black_scholes_price(100, 100, 1.0, 0.05, 0.2, 'call')
    assert abs(res.price - bs) < 0.12


def test_monte_carlo_put_close_to_black_scholes():
    res = european_option_price_mc(100, 90, 1.0, 0.05, 0.2, 150000, 'put', seed=9)
    bs = black_scholes_price(100, 90, 1.0, 0.05, 0.2, 'put')
    assert abs(res.price - bs) < 0.12

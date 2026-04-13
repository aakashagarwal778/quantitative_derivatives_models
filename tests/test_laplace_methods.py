from qdlib.transform_methods.laplace_pricing import laplace_call_price_black_scholes
from qdlib.pricing_foundations.black_scholes import black_scholes_price


def test_laplace_black_scholes_call_close_to_closed_form():
    lp = laplace_call_price_black_scholes(100, 100, 1.0, 0.03, 0.2)
    bs = black_scholes_price(100, 100, 1.0, 0.03, 0.2, 'call')
    assert abs(lp - bs) < 1e-3

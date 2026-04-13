from qdlib.pricing_foundations.black_scholes import black_scholes_price
from qdlib.pricing_foundations.parity_relations import put_call_parity_gap


def test_black_scholes_call_value():
    price = black_scholes_price(100, 100, 1.0, 0.05, 0.2, 'call')
    assert abs(price - 10.450583572185565) < 1e-8


def test_black_scholes_put_value():
    price = black_scholes_price(100, 100, 1.0, 0.05, 0.2, 'put')
    assert abs(price - 5.573526022256971) < 1e-8


def test_put_call_parity_gap_near_zero():
    S, K, T, r, sigma = 100, 110, 0.75, 0.03, 0.22
    call = black_scholes_price(S, K, T, r, sigma, 'call')
    put = black_scholes_price(S, K, T, r, sigma, 'put')
    assert abs(put_call_parity_gap(call, put, S, K, T, r)) < 1e-10

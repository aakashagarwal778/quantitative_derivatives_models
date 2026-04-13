from qdlib.pricing_foundations.black_scholes import black_scholes_price
from qdlib.pricing_foundations.implied_volatility import implied_volatility_from_price


def test_implied_volatility_recovers_sigma_for_call():
    sigma = 0.23
    price = black_scholes_price(100, 110, 1.0, 0.03, sigma, 'call')
    iv = implied_volatility_from_price(price, 100, 110, 1.0, 0.03, 'call')
    assert abs(iv - sigma) < 1e-8


def test_implied_volatility_recovers_sigma_for_put():
    sigma = 0.31
    price = black_scholes_price(95, 100, 0.8, 0.01, sigma, 'put')
    iv = implied_volatility_from_price(price, 95, 100, 0.8, 0.01, 'put')
    assert abs(iv - sigma) < 1e-8

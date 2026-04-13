from qdlib.lattice_methods.binomial_crr import binomial_option_price
from qdlib.pricing_foundations.black_scholes import black_scholes_price


def test_binomial_converges_to_black_scholes_call():
    price = binomial_option_price(100, 100, 1.0, 0.05, 0.2, 400, 'call')
    bs = black_scholes_price(100, 100, 1.0, 0.05, 0.2, 'call')
    assert abs(price - bs) < 0.03


def test_american_put_exceeds_european_put():
    euro = binomial_option_price(90, 100, 1.0, 0.05, 0.25, 300, 'put', american=False)
    amer = binomial_option_price(90, 100, 1.0, 0.05, 0.25, 300, 'put', american=True)
    assert amer >= euro

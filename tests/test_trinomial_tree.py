from qdlib.lattice_methods.trinomial_tree import trinomial_option_price
from qdlib.pricing_foundations.black_scholes import black_scholes_price


def test_trinomial_reasonable_against_black_scholes():
    tri = trinomial_option_price(100, 100, 1.0, 0.05, 0.2, 200, 'call')
    bs = black_scholes_price(100, 100, 1.0, 0.05, 0.2, 'call')
    assert abs(tri - bs) < 0.05


def test_trinomial_american_put_exceeds_european_put():
    euro = trinomial_option_price(90, 100, 1.0, 0.05, 0.25, 200, 'put', american=False)
    amer = trinomial_option_price(90, 100, 1.0, 0.05, 0.25, 200, 'put', american=True)
    assert amer >= euro

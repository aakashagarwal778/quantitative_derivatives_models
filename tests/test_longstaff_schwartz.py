from qdlib.monte_carlo.longstaff_schwartz import longstaff_schwartz_american_put
from qdlib.pricing_foundations.black_scholes import black_scholes_price


def test_longstaff_schwartz_above_european_put():
    american = longstaff_schwartz_american_put(90, 100, 1.0, 0.05, 0.25, 25000, 50, seed=5)
    european = black_scholes_price(90, 100, 1.0, 0.05, 0.25, 'put')
    assert american >= european

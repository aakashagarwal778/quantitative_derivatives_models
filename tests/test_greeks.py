import math
from qdlib.pricing_foundations.black_scholes import black_scholes_price
from qdlib.pricing_foundations.greeks import black_scholes_greeks


def test_call_gamma_positive():
    greeks = black_scholes_greeks(100, 100, 1.0, 0.05, 0.2, 'call')
    assert greeks['gamma'] > 0


def test_put_delta_negative():
    greeks = black_scholes_greeks(100, 100, 1.0, 0.05, 0.2, 'put')
    assert greeks['delta'] < 0


def test_delta_matches_finite_difference():
    S, K, T, r, sigma = 100.0, 95.0, 0.75, 0.04, 0.25
    eps = 1e-4
    fd = (black_scholes_price(S + eps, K, T, r, sigma, 'call') - black_scholes_price(S - eps, K, T, r, sigma, 'call')) / (2 * eps)
    delta = black_scholes_greeks(S, K, T, r, sigma, 'call')['delta']
    assert abs(fd - delta) < 5e-5


def test_vega_matches_finite_difference():
    S, K, T, r, sigma = 100.0, 100.0, 0.5, 0.03, 0.2
    eps = 1e-5
    fd = (black_scholes_price(S, K, T, r, sigma + eps, 'call') - black_scholes_price(S, K, T, r, sigma - eps, 'call')) / (2 * eps)
    vega = black_scholes_greeks(S, K, T, r, sigma, 'call')['vega']
    assert abs(fd - vega) < 1e-4

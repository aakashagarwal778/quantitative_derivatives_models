from qdlib.pde_ode_methods.finite_difference_explicit import explicit_fd_european_put
from qdlib.pde_ode_methods.finite_difference_crank_nicolson import crank_nicolson_european_put
from qdlib.pricing_foundations.black_scholes import black_scholes_price


def test_explicit_fd_close_to_black_scholes_put():
    fd = explicit_fd_european_put(100, 100, 1.0, 0.05, 0.2)
    bs = black_scholes_price(100, 100, 1.0, 0.05, 0.2, 'put')
    assert abs(fd - bs) < 0.2


def test_crank_nicolson_close_to_black_scholes_put():
    fd = crank_nicolson_european_put(100, 100, 1.0, 0.05, 0.2)
    bs = black_scholes_price(100, 100, 1.0, 0.05, 0.2, 'put')
    assert abs(fd - bs) < 0.08


def test_explicit_and_cn_are_close():
    e = explicit_fd_european_put(90, 100, 1.0, 0.05, 0.25)
    c = crank_nicolson_european_put(90, 100, 1.0, 0.05, 0.25)
    assert abs(e - c) < 0.2

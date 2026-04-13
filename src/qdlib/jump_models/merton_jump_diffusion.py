from __future__ import annotations

import math
from ..pricing_foundations.black_scholes import black_scholes_price


def merton_jump_diffusion_call(spot: float, strike: float, maturity: float, rate: float, volatility: float, jump_intensity: float, jump_mean: float, jump_vol: float, n_terms: int = 50) -> float:
    """Merton jump-diffusion European call price via Poisson-mixture expansion."""
    kappa = math.exp(jump_mean + 0.5 * jump_vol**2) - 1.0
    total = 0.0
    lam_t = jump_intensity * maturity
    for n in range(n_terms):
        poisson = math.exp(-lam_t) * lam_t**n / math.factorial(n)
        sigma_n = math.sqrt(volatility**2 + n * jump_vol**2 / maturity)
        r_n = rate - jump_intensity * kappa + n * jump_mean / maturity
        total += poisson * black_scholes_price(spot, strike, maturity, r_n, sigma_n, 'call')
    return total

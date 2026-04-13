from __future__ import annotations

import math
import numpy as np
from ..common.utils import PriceResult
from ..pricing_foundations.black_scholes import black_scholes_price


def european_option_price_antithetic(*args, **kwargs) -> PriceResult:
    kwargs['antithetic'] = True
    from ..monte_carlo.european_monte_carlo import european_option_price_mc
    return european_option_price_mc(*args, **kwargs)


def control_variate_call(spot: float, strike: float, maturity: float, rate: float, volatility: float, n_paths: int, dividend: float = 0.0, seed: int | None = None) -> PriceResult:
    """Price a self-quanto call using the vanilla call as a control variate."""
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(n_paths)
    terminal = spot * np.exp((rate - dividend - 0.5 * volatility**2) * maturity + volatility * np.sqrt(maturity) * z)
    discount = math.exp(-rate * maturity)
    target = discount * np.maximum(terminal**2 - strike * terminal, 0.0)
    control = discount * np.maximum(terminal - strike, 0.0)
    control_mean = black_scholes_price(spot, strike, maturity, rate, volatility, 'call', dividend)
    cov = np.cov(target, control, ddof=1)[0, 1]
    beta = cov / np.var(control, ddof=1)
    adjusted = target - beta * (control - control_mean)
    price = float(np.mean(adjusted))
    stderr = float(np.std(adjusted, ddof=1) / math.sqrt(n_paths))
    return PriceResult(price=price, stderr=stderr, ci_low=price - 1.96 * stderr, ci_high=price + 1.96 * stderr)


def importance_sampling_deep_otm_call(spot: float, strike: float, maturity: float, rate: float, volatility: float, n_paths: int, shift: float, dividend: float = 0.0, seed: int | None = None) -> PriceResult:
    """Importance sampling for a deep out-of-the-money call.

    The standard normal shock Z is sampled from N(shift, 1). The Radon-Nikodym derivative
    for the original density relative to the shifted density is exp(-shift * Y + 0.5*shift^2)
    when Y ~ N(shift, 1).
    """
    rng = np.random.default_rng(seed)
    y = rng.standard_normal(n_paths) + shift
    terminal = spot * np.exp((rate - dividend - 0.5 * volatility**2) * maturity + volatility * np.sqrt(maturity) * y)
    likelihood_ratio = np.exp(-shift * y + 0.5 * shift**2)
    discounted = math.exp(-rate * maturity) * np.maximum(terminal - strike, 0.0) * likelihood_ratio
    price = float(np.mean(discounted))
    stderr = float(np.std(discounted, ddof=1) / math.sqrt(n_paths))
    return PriceResult(price=price, stderr=stderr, ci_low=price - 1.96 * stderr, ci_high=price + 1.96 * stderr)

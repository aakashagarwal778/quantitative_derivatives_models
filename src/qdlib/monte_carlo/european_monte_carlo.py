from __future__ import annotations

import math
import numpy as np
from ..common.utils import PriceResult
from ..monte_carlo.gbm_simulation import simulate_gbm_terminal


def european_option_price_mc(spot: float, strike: float, maturity: float, rate: float, volatility: float, n_paths: int, option_type: str = 'call', dividend: float = 0.0, seed: int | None = None, antithetic: bool = False) -> PriceResult:
    terminal = simulate_gbm_terminal(spot, maturity, rate, volatility, n_paths, dividend, seed, antithetic)
    if option_type.lower() == 'call':
        payoff = np.maximum(terminal - strike, 0.0)
    elif option_type.lower() == 'put':
        payoff = np.maximum(strike - terminal, 0.0)
    else:
        raise ValueError("option_type must be 'call' or 'put'.")
    disc = math.exp(-rate * maturity)
    discounted = disc * payoff
    price = float(np.mean(discounted))
    stderr = float(np.std(discounted, ddof=1) / math.sqrt(n_paths))
    return PriceResult(price=price, stderr=stderr, ci_low=price - 1.96 * stderr, ci_high=price + 1.96 * stderr)

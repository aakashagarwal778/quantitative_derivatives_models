from __future__ import annotations

from ..common.root_finding import solve_scalar_root
from ..pricing_foundations.black_scholes import black_scholes_price


def implied_volatility_from_price(price: float, spot: float, strike: float, maturity: float, rate: float, option_type: str = 'call', dividend: float = 0.0, lower: float = 1e-6, upper: float = 5.0) -> float:
    """Invert a Black–Scholes option price to its implied volatility."""
    def objective(vol: float) -> float:
        return black_scholes_price(spot, strike, maturity, rate, vol, option_type, dividend) - price
    return solve_scalar_root(objective, lower, upper)

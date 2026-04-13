from __future__ import annotations

import math


def sabr_black_implied_vol(forward: float, strike: float, maturity: float, alpha: float, beta: float, rho: float, nu: float) -> float:
    if forward <= 0 or strike <= 0:
        raise ValueError('forward and strike must be positive')
    if abs(forward - strike) < 1e-12:
        fk = forward ** (1 - beta)
        term1 = alpha / fk
        term2 = 1 + (((1 - beta)**2 / 24) * (alpha**2 / fk**2) + (rho * beta * nu * alpha) / (4 * fk) + (2 - 3 * rho**2) * nu**2 / 24) * maturity
        return term1 * term2
    log_fk = math.log(forward / strike)
    fk_beta = (forward * strike) ** ((1 - beta) / 2)
    z = (nu / alpha) * fk_beta * log_fk
    xz = math.log((math.sqrt(1 - 2 * rho * z + z*z) + z - rho) / (1 - rho))
    denom = fk_beta * (1 + ((1 - beta)**2 / 24) * log_fk**2 + ((1 - beta)**4 / 1920) * log_fk**4)
    corr = 1 + (((1 - beta)**2 / 24) * (alpha**2 / fk_beta**2) + (rho * beta * nu * alpha) / (4 * fk_beta) + (2 - 3 * rho**2) * nu**2 / 24) * maturity
    return (alpha / denom) * (z / xz) * corr

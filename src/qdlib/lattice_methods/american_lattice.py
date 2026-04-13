from __future__ import annotations

from ..lattice_methods.binomial_crr import binomial_option_price


def american_option_price_crr(*args, **kwargs) -> float:
    kwargs['american'] = True
    return binomial_option_price(*args, **kwargs)

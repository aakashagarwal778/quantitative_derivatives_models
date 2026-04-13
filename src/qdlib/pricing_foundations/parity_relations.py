from __future__ import annotations

import math


def put_call_parity_gap(call_price: float, put_price: float, spot: float, strike: float, maturity: float, rate: float, dividend: float = 0.0) -> float:
    return call_price - put_price - (spot * math.exp(-dividend * maturity) - strike * math.exp(-rate * maturity))

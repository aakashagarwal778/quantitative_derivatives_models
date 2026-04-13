from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable


def validate_positive(**kwargs: float) -> None:
    for name, value in kwargs.items():
        if value <= 0:
            raise ValueError(f"{name} must be positive; got {value}.")


def validate_probability(name: str, value: float) -> None:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must lie in [0, 1]; got {value}.")


def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


@dataclass(frozen=True)
class PriceResult:
    price: float
    stderr: float | None = None
    ci_low: float | None = None
    ci_high: float | None = None



def mean(values: Iterable[float]) -> float:
    vals = list(values)
    if not vals:
        raise ValueError("values must be non-empty")
    return sum(vals) / len(vals)

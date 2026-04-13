from __future__ import annotations

import numpy as np
import pandas as pd


def compute_log_returns(prices: pd.Series) -> pd.Series:
    prices = prices.astype(float)
    return np.log(prices / prices.shift(1)).dropna()


def annualized_mean_vol(log_returns: pd.Series, periods_per_year: int = 252) -> tuple[float, float]:
    mu = float(log_returns.mean() * periods_per_year)
    sigma = float(log_returns.std(ddof=1) * np.sqrt(periods_per_year))
    return mu, sigma

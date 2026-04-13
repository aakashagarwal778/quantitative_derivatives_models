from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _bootstrap import *  # noqa: F401,F403

from qdlib.pricing_foundations.black_scholes import black_scholes_price

params=dict(spot=100,strike=100,maturity=1.0,rate=0.05,volatility=0.2)
print("Call:", round(black_scholes_price(**params, option_type="call"),6))
print("Put :", round(black_scholes_price(**params, option_type="put"),6))

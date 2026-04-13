from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _bootstrap import *  # noqa: F401,F403

from qdlib.lattice_methods.binomial_crr import binomial_option_price
from qdlib.pricing_foundations.black_scholes import black_scholes_price

params=dict(spot=100,strike=100,maturity=1.0,rate=0.05,volatility=0.2,option_type="call")
bs=black_scholes_price(**params)
crr=binomial_option_price(**params, steps=500)
print("Black-Scholes:", round(bs,6))
print("CRR 500-step :", round(crr,6))

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _bootstrap import *  # noqa: F401,F403

from qdlib.lattice_methods.binomial_crr import binomial_option_price
from qdlib.lattice_methods.trinomial_tree import trinomial_option_price

args=dict(spot=100,strike=100,maturity=1.0,rate=0.05,volatility=0.2,option_type="call")
print("CRR      :", round(binomial_option_price(**args, steps=300),6))
print("Trinomial:", round(trinomial_option_price(**args, steps=150),6))

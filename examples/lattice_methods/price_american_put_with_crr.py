from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _bootstrap import *  # noqa: F401,F403

from qdlib.lattice_methods.american_lattice import american_option_price_crr

price=american_option_price_crr(spot=100,strike=100,maturity=1.0,rate=0.05,volatility=0.2,steps=500,option_type="put")
print("American put price:", round(price,6))

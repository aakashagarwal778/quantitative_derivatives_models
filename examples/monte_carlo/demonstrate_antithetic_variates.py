from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _bootstrap import *  # noqa: F401,F403

from qdlib.monte_carlo.european_monte_carlo import european_option_price_mc
from qdlib.monte_carlo.variance_reduction import european_option_price_antithetic

plain=european_option_price_mc(100,100,1.0,0.05,0.2,20000,"call",seed=1)
ant=european_option_price_antithetic(100,100,1.0,0.05,0.2,20000,"call",seed=1)
print("Plain     ", plain)
print("Antithetic", ant)

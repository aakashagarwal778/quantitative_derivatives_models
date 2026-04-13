from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _bootstrap import *  # noqa: F401,F403

from qdlib.monte_carlo.european_monte_carlo import european_option_price_mc

res=european_option_price_mc(100,100,1.0,0.05,0.2,50000,"call",seed=42)
print(res)

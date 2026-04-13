from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _bootstrap import *  # noqa: F401,F403

from qdlib.stochastic_volatility.heston_implied_volatility import heston_implied_volatility
for k in [80,90,100,110,120]:
    print(k, round(heston_implied_volatility(100,k,1.0,0.02,1.5,0.04,0.4,-0.6,0.04),6))

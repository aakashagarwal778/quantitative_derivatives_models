from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _bootstrap import *  # noqa: F401,F403

from qdlib.stochastic_volatility.heston_fft import heston_price_grid
print(heston_price_grid([90,100,110],100,1.0,0.02,1.5,0.04,0.4,-0.6,0.04))

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _bootstrap import *  # noqa: F401,F403

from qdlib.sabr.sabr_implied_volatility import sabr_black_implied_vol
for k in [80,90,100,110,120]:
    print(k, round(sabr_black_implied_vol(100,k,1.0,0.2,0.5,-0.2,0.6),6))

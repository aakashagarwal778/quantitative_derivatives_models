from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _bootstrap import *  # noqa: F401,F403

from qdlib.transform_methods.laplace_pricing import laplace_call_price_black_scholes
print(round(laplace_call_price_black_scholes(100,100,1.0,0.05,0.2),6))

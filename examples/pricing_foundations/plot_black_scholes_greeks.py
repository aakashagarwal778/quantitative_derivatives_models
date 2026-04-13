from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _bootstrap import *  # noqa: F401,F403

from qdlib.pricing_foundations.greeks import black_scholes_greeks

g=black_scholes_greeks(100,100,1.0,0.05,0.2,"call")
for k,v in g.items():
    print(f"{k}: {v:.6f}")

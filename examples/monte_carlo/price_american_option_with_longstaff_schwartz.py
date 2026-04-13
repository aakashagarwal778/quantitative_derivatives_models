from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _bootstrap import *  # noqa: F401,F403

from qdlib.monte_carlo.longstaff_schwartz import longstaff_schwartz_american_put
print(round(longstaff_schwartz_american_put(100,100,1.0,0.05,0.2,20000,50,seed=42),6))

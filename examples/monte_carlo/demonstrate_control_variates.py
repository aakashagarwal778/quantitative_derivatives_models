from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _bootstrap import *  # noqa: F401,F403

from qdlib.monte_carlo.variance_reduction import control_variate_call
print(control_variate_call(100,100,1.0,0.05,0.2,50000,seed=42))

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _bootstrap import *  # noqa: F401,F403

from qdlib.monte_carlo.variance_reduction import importance_sampling_deep_otm_call
print(importance_sampling_deep_otm_call(100,140,1.0,0.05,0.2,50000,shift=1.5,seed=42))

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _bootstrap import *  # noqa: F401,F403

import numpy as np
from qdlib.local_volatility.local_vol_surface import LocalVolSurface

surface=LocalVolSurface(np.array([0.5,1.0,2.0]), np.array([80,100,120]), np.array([[0.24,0.2,0.22],[0.25,0.21,0.23],[0.26,0.22,0.24]]))
print(surface(1.25,110))

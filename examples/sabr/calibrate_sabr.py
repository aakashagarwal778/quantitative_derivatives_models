from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _bootstrap import *  # noqa: F401,F403

import numpy as np
from qdlib.sabr.sabr_calibration import calibrate_sabr
strikes=np.array([80,90,100,110,120],dtype=float)
vols=np.array([0.29,0.25,0.21,0.22,0.25],dtype=float)
print(calibrate_sabr(strikes,vols,100,1.0,0.5).x)

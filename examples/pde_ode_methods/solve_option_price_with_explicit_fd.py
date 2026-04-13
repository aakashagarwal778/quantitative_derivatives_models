from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _bootstrap import *  # noqa: F401,F403

from qdlib.pde_ode_methods.finite_difference_explicit import explicit_fd_european_put
print(round(explicit_fd_european_put(100,100,1.0,0.05,0.2),6))

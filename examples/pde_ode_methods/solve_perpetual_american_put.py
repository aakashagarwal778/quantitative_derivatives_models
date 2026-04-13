from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _bootstrap import *  # noqa: F401,F403

from qdlib.pde_ode_methods.perpetual_american_put_ode import perpetual_american_put_price
print(round(perpetual_american_put_price(120,100,0.05,0.2),6))

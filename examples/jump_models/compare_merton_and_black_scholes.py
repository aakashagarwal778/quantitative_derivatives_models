from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _bootstrap import *  # noqa: F401,F403

from qdlib.jump_models.merton_jump_diffusion import merton_jump_diffusion_call
from qdlib.pricing_foundations.black_scholes import black_scholes_price

bs=black_scholes_price(100,100,1.0,0.05,0.2,"call")
mj=merton_jump_diffusion_call(100,100,1.0,0.05,0.2,0.6,-0.1,0.3)
print("BS    :", round(bs,6))
print("Merton:", round(mj,6))

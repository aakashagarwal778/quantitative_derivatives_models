from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _bootstrap import *  # noqa: F401,F403

from qdlib.pricing_foundations.implied_volatility import implied_volatility_from_price

vol=implied_volatility_from_price(10.450583572185565,100,100,1.0,0.05,"call")
print("Implied vol:", round(vol,6))

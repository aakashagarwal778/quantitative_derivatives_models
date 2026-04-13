from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _bootstrap import *  # noqa: F401,F403

import pandas as pd
from pathlib import Path
from qdlib.calibration.heston_calibration import calibrate_heston

path=Path(__file__).resolve().parents[2]/"data/sample_data/sp500_option_prices.csv"
df=pd.read_csv(path)
res=calibrate_heston(100,df.strike,df.maturity,df.rate,df.price)
print(res.x)

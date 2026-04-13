from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _bootstrap import *  # noqa: F401,F403

import pandas as pd
from pathlib import Path
from qdlib.empirical.log_returns import compute_log_returns, annualized_mean_vol
path=Path(__file__).resolve().parents[2]/"data/sample_data/dax_2024.csv"
df=pd.read_csv(path)
rets=compute_log_returns(df["close"])
print(annualized_mean_vol(rets))

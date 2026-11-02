"""Row subsets of an activation dir (acts.pt + positions.parquet + args.json) for p3_readout.py --src.

  python q38_subset.py --acts runs/arm1_P2 --out q38/sample36 --mode sample   # 2,000 random rows (seed 0): FVE gate
  python q38_subset.py --acts q38/acts38_P1 --out q38/acts38_P1h --mode half   # every other cached position per transcript

"half" keeps the 1st, 3rd, 5th, ... of each transcript's (sorted) cached positions: 25 of 50, the same rows for any
model, so a readout pairs with Control1's graded step-1000 explanations at those (tid, pos).
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--acts", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--mode", choices=["sample", "half"], required=True)
ap.add_argument("--n", type=int, default=2000)
a = ap.parse_args()
pos = pd.read_parquet(f"{a.acts}/positions.parquet").reset_index(drop=True)
if a.mode == "sample":
    rows = np.sort(np.random.default_rng(0).choice(len(pos), a.n, replace=False))
else:
    rows = np.flatnonzero(pos.groupby("tid").pos.rank(method="first").to_numpy() % 2 == 1)
os.makedirs(a.out, exist_ok=True)
torch.save(torch.load(f"{a.acts}/acts.pt")[torch.from_numpy(rows)].clone(), f"{a.out}/acts.pt")
pos.iloc[rows].reset_index(drop=True).to_parquet(f"{a.out}/positions.parquet")
json.dump(dict(json.load(open(f"{a.acts}/args.json")), out=a.out, subset=a.mode, subset_of=a.acts),
          open(f"{a.out}/args.json", "w"), indent=1)
print(f"{a.out}: {len(rows)} of {len(pos)} rows", flush=True)

"""INVESTIGATION_20261017_NOISE_LEVELS.md: merge the explain shards of one noise readout (pulled from one or more boxes)
with its source positions into runs/<name>/readouts.parquet, in the same row order and columns as p3_readout.py merge
without an AR (no mse/cos), so grade_runs.py treats it like any other run.

  python noise_merge.py --parts tmp/noise/pull/l2r1_P1 --src runs/arm1_P1 --n_workers 6 --out runs/noise_l2r1_P1
"""
import argparse
import glob
import json
import os

import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--parts", required=True)
ap.add_argument("--src", required=True)
ap.add_argument("--n_workers", type=int, required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()

pos = pd.read_parquet(f"{a.src}/positions.parquet").reset_index(drop=True)
files = sorted(glob.glob(f"{a.parts}/w*.parquet"))
assert len(files) == a.n_workers, (len(files), a.n_workers)
ex = pd.concat([pd.read_parquet(f) for f in files]).set_index("row").sort_index()
assert list(ex.index) == list(range(len(pos))), "missing rows"
df = pd.concat([pos, ex.reset_index(drop=True)], axis=1)
os.makedirs(a.out, exist_ok=True)
df.to_parquet(f"{a.out}/readouts.parquet")
src_args = json.load(open(f"{a.src}/args.json"))
json.dump(dict(src_args, out=a.out, nla="ljhunt/nla-qwen36-27b-p3-control1 checkpoints/rl_final/av", ar=None, no_ar=True),
          open(f"{a.out}/args.json", "w"), indent=1)
print(f"{a.out}: {len(df)} rows, closed {df.closed.mean():.3f}")

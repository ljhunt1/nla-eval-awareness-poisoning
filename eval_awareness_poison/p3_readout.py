"""Phase 3 eval-awareness readouts of one NLA checkpoint on the Phase 2 measurement set (P1 + P2), using the SAME cached
layer-42 activations and seeded positions as every Phase 2 arm (runs/arm1_{P1,P2}/acts.pt + positions.parquet), so
explanations pair across NLAs and checkpoints. Explanation = readout.explain (T=1, no top-k/top-p, 200 new tokens);
FVE = readout.reconstruct with the checkpoint's own AR. METHODS_EVAL_AWARENESS.md §1-2.

  python p3_readout.py explain --av DIR --src DIR --out DIR --worker k --n_workers n   # one GPU, shard k of n
  python p3_readout.py merge   [--ar DIR] --src DIR --out DIR --n_workers n            # concat shards (+ AR FVE) -> readouts.parquet
"""
import argparse
import glob
import json
import os
import types

import pandas as pd
import torch

import readout as R


def cmd_explain(a):
    V = torch.load(f"{a.src}/acts.pt").float()
    idx = list(range(len(V)))[a.worker::a.n_workers]
    f = f"{a.out}/parts/w{a.worker:02d}.parquet"
    if os.path.exists(f):
        return
    os.makedirs(f"{a.out}/parts", exist_ok=True)
    ex = R.explain(types.SimpleNamespace(bs=a.bs, max_new=a.max_new), V[idx], {"av": a.av})
    ex.insert(0, "row", idx)
    ex.to_parquet(f + ".tmp")
    os.replace(f + ".tmp", f)


def cmd_merge(a):
    pos = pd.read_parquet(f"{a.src}/positions.parquet").reset_index(drop=True)
    parts = [pd.read_parquet(f) for f in sorted(glob.glob(f"{a.out}/parts/w*.parquet"))]
    assert len(parts) == a.n_workers, (len(parts), a.n_workers)
    ex = pd.concat(parts).set_index("row").sort_index()
    assert list(ex.index) == list(range(len(pos))), "missing rows"
    df = pd.concat([pos, ex.reset_index(drop=True)], axis=1)
    fve = rawvar = None
    if a.ar:  # without --ar: explanations only, no FVE
        V = torch.load(f"{a.src}/acts.pt").float()
        mse, cos, rawvar, _ = R.reconstruct(df.explanation.tolist(), V, {"ar": a.ar})
        df["mse"], df["cos"] = mse, cos
        fve = float(1 - mse.mean() / rawvar)
    df.to_parquet(f"{a.out}/readouts.parquet")
    src_args = json.load(open(f"{a.src}/args.json"))
    json.dump(dict(src_args, out=a.out, nla=a.av, ar=a.ar, rawvar=rawvar, fve=fve),
              open(f"{a.out}/args.json", "w"), indent=1)
    print(f"{a.out}: {len(df)} rows, closed {df.closed.mean():.3f}, FVE {fve}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["explain", "merge"])
    ap.add_argument("--av"); ap.add_argument("--ar"); ap.add_argument("--src", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--worker", type=int, default=0); ap.add_argument("--n_workers", type=int, default=1)
    ap.add_argument("--bs", type=int, default=64); ap.add_argument("--max_new", type=int, default=200)
    a = ap.parse_args()
    {"explain": cmd_explain, "merge": cmd_merge}[a.cmd](a)

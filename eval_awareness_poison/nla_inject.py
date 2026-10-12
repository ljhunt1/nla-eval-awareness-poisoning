"""Explain saved activations after adding a direction, without re-running the target (does the NLA verbalize it?).

  python nla_inject.py --run runs/gemma12b_belief_rand --group belief_bottom --vec steer/belief/probe_vec_L32.pt \
      --layer 32 --mags 550,1800,5500 --nla gemma12b --out runs/gemma12b_inject_belief

Adds mag * unit(vec[layer]) to every saved activation of the chosen transcripts and writes one run directory per
magnitude (<out>_<mag>/readouts.parquet + args.json), gradeable with grade.py.
"""

import argparse
import json
import os
from types import SimpleNamespace

import pandas as pd
import torch

from readout import NLAS, explain


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--group", required=True)
    ap.add_argument("--vec", required=True)
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--mags", required=True)
    ap.add_argument("--nla", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--bs", type=int, default=128)
    a = ap.parse_args()
    pos = pd.read_parquet(f"{a.run}/positions.parquet")
    V = torch.load(f"{a.run}/acts.pt").float()
    keep = (pos.group == a.group).values
    pos, V = pos[keep].reset_index(drop=True), V[torch.tensor(keep)]
    u = torch.load(a.vec)[a.layer].float()
    u = u / u.norm()
    src = json.load(open(f"{a.run}/args.json"))
    for mag in [float(x) for x in a.mags.split(",")]:
        out = f"{a.out}_{mag:g}"
        os.makedirs(out, exist_ok=True)
        ex = explain(SimpleNamespace(bs=a.bs, max_new=200), V + mag * u, NLAS[a.nla])
        pd.concat([pos, ex], axis=1).to_parquet(f"{out}/readouts.parquet")
        json.dump({**src, "inject_vec": a.vec, "inject_mag": mag, "group": a.group}, open(f"{out}/args.json", "w"))
        print("wrote", out, flush=True)


if __name__ == "__main__":
    main()

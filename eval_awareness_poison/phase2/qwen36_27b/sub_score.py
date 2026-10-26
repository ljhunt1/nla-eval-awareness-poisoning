"""Substitution feasibility check (PHASE35_AV_SELECTIVE.md, follow-ups): AR reconstruction error for given explanations.
Single GPU.   python sub_score.py --ar <ar> --bat <dir with acts_<src>.pt> --meta <meta.json> --inp X.parquet --out Y.parquet
X.parquet rows: src, ref, expl (plus anything else, kept). Y adds mse (same normalisation as avs_mon.py) and ntok.
"""
import argparse, json
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer
from nla36 import Critic, ar_batch, normalize


def main():
    p = argparse.ArgumentParser()
    for k in ["ar", "bat", "meta", "inp", "out"]:
        p.add_argument(f"--{k}", required=True)
    p.add_argument("--bs", type=int, default=32)
    a = p.parse_args()
    meta = json.load(open(a.meta))
    tok = AutoTokenizer.from_pretrained(meta["model_path"])
    d = pd.read_parquet(a.inp).reset_index(drop=True)
    acts = {s: torch.load(f"{a.bat}/acts_{s}.pt") for s in d.src.unique()}
    V = torch.stack([acts[s][r] for s, r in zip(d.src, d.ref)])
    critic = Critic(a.ar, meta["layer"] + 1).cuda().eval()
    s_ = meta["mse_scale"]
    mse = np.full(len(d), np.nan)
    with torch.no_grad():
        for b in range(0, len(d), a.bs):
            j = list(range(b, min(b + a.bs, len(d))))
            pr = critic(*ar_batch(tok, [d.expl[k] for k in j]))
            mse[j] = ((normalize(pr, s_) - normalize(V[j].cuda(), s_)) ** 2).mean(-1).cpu().numpy()
    d["mse"] = mse
    d["ntok"] = [len(tok(e, add_special_tokens=False).input_ids) for e in d.expl]
    d.to_parquet(a.out)
    print("scored", len(d), "mean mse", float(np.nanmean(mse)))


if __name__ == "__main__":
    main()

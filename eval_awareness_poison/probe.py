"""Linear eval-vs-deploy probe at the NLA layer, and how much of it survives the NLA round trip.

  python probe.py --run runs/qwen7b_T --steer steer/qwen7b/avg.pt --layer 20

Token features = target activations at the 50 sampled response tokens (acts.pt) or the AR's reconstructions of
them (recon.pt). Labels = transcript eval/deploy. 5-fold CV grouped by transcript. Reports token-level AUC and
transcript-level AUC (mean probe score over the transcript's tokens), for: probe on true activations, the same
probe applied to reconstructions, a probe trained on reconstructions, and the projection onto the steering vector
at the NLA layer (no training).
"""

import argparse
import json

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold


def unit(x):
    x = np.nan_to_num(x, nan=0.0, posinf=6.5e4, neginf=-6.5e4)  # runs saved in fp16 overflow on Gemma outlier dims
    return x / np.linalg.norm(x, axis=1, keepdims=True).clip(1e-9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--steer", default=None)
    ap.add_argument("--layer", type=int, default=20)
    ap.add_argument("--C", type=float, default=0.01)
    a = ap.parse_args()
    df = pd.read_parquet(f"{a.run}/readouts.parquet")
    X = unit(torch.load(f"{a.run}/acts.pt").float().numpy())
    R = unit(torch.load(f"{a.run}/recon.pt").float().numpy())
    y = (df.label == "eval").to_numpy().astype(int)
    groups = df.tid.to_numpy()
    out = {}
    scores = {k: np.zeros(len(y)) for k in ["true", "true_probe_on_recon", "recon_probe_on_recon"]}
    for tr, te in GroupKFold(5).split(X, y, groups):
        p = LogisticRegression(C=a.C, max_iter=2000).fit(X[tr], y[tr])
        scores["true"][te] = p.decision_function(X[te])
        scores["true_probe_on_recon"][te] = p.decision_function(R[te])
        q = LogisticRegression(C=a.C, max_iter=2000).fit(R[tr], y[tr])
        scores["recon_probe_on_recon"][te] = q.decision_function(R[te])
    if a.steer:
        v = torch.load(a.steer)[a.layer].float().numpy()
        v = v / np.linalg.norm(v)
        scores["steer_proj_true"] = X @ v
        scores["steer_proj_recon"] = R @ v
    for k, s in scores.items():
        t = pd.DataFrame({"tid": groups, "y": y, "s": s}).groupby("tid").agg(y=("y", "first"), s=("s", "mean"))
        out[k] = {"token_auc": roc_auc_score(y, s), "transcript_auc": roc_auc_score(t.y, t.s)}
        print(f"{k:24s} token AUC {out[k]['token_auc']:.3f}   transcript AUC {out[k]['transcript_auc']:.3f}")
    # per-group mean transcript score for the true probe (which groups look most eval-like)
    t = pd.DataFrame({"tid": groups, "group": df.group, "label": df.label, "s": scores["true"]}).groupby(
        ["label", "group", "tid"]).s.mean().groupby(["label", "group"]).mean()
    print(t.round(2).to_string())
    json.dump(out, open(f"{a.run}/probe.json", "w"), indent=1)


if __name__ == "__main__":
    main()

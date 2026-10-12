"""Does Gemma hold a latent, transcript-specific eval belief before anyone asks? (no NLA involved)

  python belief_probe.py runs/gemma12b_belief runs/gemma12b_credence.parquet data/H_gemma12b.jsonl [steer/gemma12b/avg.pt]

Target: the model's later answer to "evaluation or real?" (credence.py, logP difference), asked *after* the
transcript. Features: activations on the transcript itself (belief_acts.py), before the question exists.
- (a) eval awareness in general: does credence separate eval from deploy at all?
- (b) unverbalized eval belief: do pre-question activations predict credence *within source* (residualized on
  source means and transcript length), out of fold? The responses almost never verbalize (0.3% in resampling), so a
  within-source signal is a belief the model holds but hasn't said.
- (c) specific transcripts: the highest out-of-fold predicted belief (within-source residual) among transcripts
  whose final turn doesn't verbalize, written to <run>/candidates.jsonl for the NLA.
Also: does the eval-awareness steering direction (causal for credence) track credence within source?
"""

import json
import re
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import RidgeCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from nearverb import VERB


def resid(v, df):
    """Remove source means and a linear log-length trend from the values v (aligned with df)."""
    v = np.asarray(v, dtype=float)
    r = v - pd.Series(v).groupby(df.source.values).transform("mean").values
    X = np.c_[np.ones(len(df)), np.log(df.n_tok.values)]
    b, *_ = np.linalg.lstsq(X, r, rcond=None)
    return r - X @ b


def main(run, cred_path, transcripts, steer=None):
    A = torch.load(f"{run}/acts.pt")
    cred = pd.read_parquet(cred_path).set_index("tid")
    rows = {r["id"]: r for r in map(json.loads, open(transcripts))}
    df = pd.DataFrame({"tid": A["tid"]})
    df = df.join(cred[["label", "group", "source", "credence", "n_tok"]], on="tid")
    keep = df.credence.notna().values
    df = df[keep].reset_index(drop=True)
    df["verb"] = df.tid.map(lambda t: bool(VERB.search(rows[t]["messages"][-1].get("content") or "")))
    df["agent"] = df.tid.map(lambda t: "agent" in rows[t].get("blurb", ""))
    y = df.credence
    print(f"(a) {len(df)} transcripts. credence eval-vs-deploy AUC {roc_auc_score(df.label == 'eval', y):.3f}; "
          f"agentic {roc_auc_score(df[df.agent].label == 'eval', y[df.agent]):.3f}")
    yr = resid(y, df)
    print(f"    credence variance explained by source + length: {1 - yr.var() / y.var():.2f}")
    kf = KFold(5, shuffle=True, random_state=0)
    best = None
    print("(b) out-of-fold prediction of credence from pre-question activations")
    print("    pool        layer   r(all)   r(within source, length-residualized)")
    for pool in ["mean_all", "mean_final", "last"]:
        X = A[pool].float().numpy()[keep]
        for j, L in enumerate(A["layers"]):
            m = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(1, 5, 9)))
            p = cross_val_predict(m, X[:, j], y.values, cv=kf)
            pr = resid(p, df)
            rw = np.corrcoef(pr, yr)[0, 1]
            print(f"    {pool:10s}  {L:5d}   {np.corrcoef(p, y)[0, 1]:.3f}    {rw:.3f}")
            if best is None or rw > best[0]:
                best = (rw, pool, L, p, pr)
    # a stricter version of (b): residualize the *target* on source/length too, and fit the probe on that
    rw, pool, L, p, pr = best
    j = A["layers"].index(L)
    X = A[pool].float().numpy()[keep][:, j]
    m = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(1, 5, 9)))
    p2 = cross_val_predict(m, X, yr, cv=kf)
    print(f"    best {pool} L{L}; probe trained on residualized credence: out-of-fold r = "
          f"{np.corrcoef(p2, yr)[0, 1]:.3f}")
    if steer:
        V = torch.load(steer)
        for jj, LL in enumerate(A["layers"]):
            u = V[LL].float().numpy()
            Xm = A["mean_all"].float().numpy()[keep][:, jj]
            proj = (Xm - Xm.mean(0)) @ (u / np.linalg.norm(u))
            pr2 = resid(proj, df)
            print(f"    steering-direction projection L{LL}: r(all) {np.corrcoef(proj, y)[0, 1]:.3f}, "
                  f"r(within) {np.corrcoef(pr2, yr)[0, 1]:.3f}")
    # (c) candidates: highest within-source predicted belief, eval or deploy, final turn not verbalizing
    df["belief_resid"] = p2
    c = df[~df.verb].sort_values("belief_resid", ascending=False)
    print("(c) top candidates by within-source predicted belief:")
    print(c.head(15)[["tid", "label", "source", "credence", "belief_resid"]].to_string(index=False))
    c.to_parquet(f"{run}/belief_scores.parquet")


if __name__ == "__main__":
    main(*sys.argv[1:])

"""Analysis for INVESTIGATION_20261026_QWEN38_27B.md: Qwen3.8-27B vs Qwen3.6-27B on the Phase 2 measurement set (P1 + P2).
Every comparison holds the transcript text fixed (Qwen3.6's replies) except test 1, where each model writes its own.

  python q38_analyze.py verbal    # test 1: verbalized eval awareness of each model's own replies (same prompts, same grader)
  python q38_analyze.py cred      # test 2: "evaluation or real?" logit, eval-vs-deploy AUROC
  python q38_analyze.py acts      # tests 3+5: layer-42 eval-vs-deploy probe, gap sizes, steering-direction separation
  python q38_analyze.py weights   # test 4a: per-tensor weight distance 3.6 -> 3.8
  python q38_analyze.py nla       # test 4b: Control1 (Qwen3.6 NLA) read zero-shot on Qwen3.8 activations
Inputs under q38/ (pulled from the boxes) and runs/.
"""
import glob
import json
import sys

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import roc_auc_score

EV, DEP = "eval", "deploy"


def mcn(a, b):
    """Exact McNemar on paired booleans: (a only, b only, p)."""
    x, y = int((a & ~b).sum()), int((~a & b).sum())
    return x, y, (binomtest(x, x + y, 0.5).pvalue if x + y else 1.0)


def boot_auc_diff(y, s1, s2, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    y, s1, s2 = map(np.asarray, (y, s1, s2))
    d = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() == y[i].max():
            continue
        d.append(roc_auc_score(y[i], s2[i]) - roc_auc_score(y[i], s1[i]))
    return np.percentile(d, [2.5, 97.5])


def tx_meta():
    rows = [json.loads(l) for f in ("data/P1_qwen36_27b.jsonl", "data/P2_qwen36_27b.jsonl") for l in open(f)]
    return pd.DataFrame([dict(tid=r["id"], label=r["label"], group=r["group"]) for r in rows]).drop_duplicates("tid")


def verbal():
    v36 = pd.concat([pd.read_parquet("runs/verbalized_phase1.parquet").query("file.str.contains('qwen36_27b')", engine="python"),
                     pd.read_parquet("runs/verbalized_P2.parquet")]).drop_duplicates("tid").set_index("tid").verbalized
    for eff in ("medium", "xhigh"):
        f = f"q38/verbalized_{eff}.parquet"
        if not glob.glob(f):
            continue
        v = pd.read_parquet(f).set_index("tid")
        gen = pd.concat([pd.read_json(g, lines=True) for g in sorted(glob.glob(f"q38/gen/{eff}_w*.jsonl"))]).set_index("id")
        v["v38"] = v.verbalized.fillna(False).astype(bool)
        v["v36"] = v36.reindex(v.index).fillna(False).astype(bool)
        v["len38"] = gen.n_tok.reindex(v.index)
        v["trunc38"] = gen.finish.reindex(v.index).eq("length")
        print(f"\n## verbalized eval awareness, Qwen3.8 {eff} vs Qwen3.6 (same prompts, same grader; paired McNemar)")
        out = []
        for (lab, grp), g in list(v.groupby(["label", "group"])) + [((l, "ALL"), g) for l, g in v.groupby("label")]:
            a, b, p = mcn(g.v36, g.v38)
            out.append(dict(label=lab, group=grp, n=len(g), q36=int(g.v36.sum()), q38=int(g.v38.sum()), q36_only=a,
                            q38_only=b, p=round(p, 4), q38_tok_median=int(g.len38.median()), q38_trunc=int(g.trunc38.sum())))
        print(pd.DataFrame(out).to_string(index=False))
        print("grader:", v.grader.value_counts().to_dict(), "unparsed:", int(v.verbalized.isna().sum()))


def cred():
    m = tx_meta().set_index("tid")
    c = {M: pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f"q38/cred{M}_P*_w*.parquet"))]).set_index("tid").credence
         for M in ("36", "38")}
    d = m.join(pd.DataFrame({"c36": c["36"], "c38": c["38"]})).dropna()
    d = d[d.label.isin([EV, DEP])]
    y = (d.label == EV).to_numpy()
    a36, a38 = roc_auc_score(y, d.c36), roc_auc_score(y, d.c38)
    lo, hi = boot_auc_diff(y, d.c36, d.c38)
    print(f"\n## 'evaluation or real?' (logP Evaluation - logP Real), {len(d)} transcripts scored by both")
    print(f"eval-vs-deploy AUROC: Qwen3.6 {a36:.3f}, Qwen3.8 {a38:.3f}; difference {a38 - a36:+.3f} [95% boot {lo:+.3f}, {hi:+.3f}]")
    print(f"P(answer Evaluation) median: 3.6 eval {np.median(1 / (1 + np.exp(-d.c36[y]))):.2f} / deploy {np.median(1 / (1 + np.exp(-d.c36[~y]))):.2f}; "
          f"3.8 eval {np.median(1 / (1 + np.exp(-d.c38[y]))):.2f} / deploy {np.median(1 / (1 + np.exp(-d.c38[~y]))):.2f}")
    print(d.groupby(["label", "group"])[["c36", "c38"]].mean().round(2).to_string())
    print(f"rank correlation 3.6 vs 3.8 across transcripts: {d.c36.corr(d.c38, method='spearman'):.2f}")


def acts():
    import torch
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    res = {}
    for M, srcs in (("36", ["runs/arm1_P1", "runs/arm1_P2"]), ("38", ["q38/acts38_P1", "q38/acts38_P2"])):
        pos = pd.concat([pd.read_parquet(f"{s}/positions.parquet") for s in srcs]).reset_index(drop=True)
        V = torch.cat([torch.load(f"{s}/acts.pt").float() for s in srcs])
        steer = torch.load(f"q38/steer{M}/avg.pt").float()[42]
        u = steer / steer.norm()
        tn = V.norm(dim=-1)
        pos["proj"] = ((V - V.mean(0)) @ u).numpy() / tn.mean().item()
        keep = pos.label.isin([EV, DEP]).to_numpy()
        ev = torch.from_numpy((pos.label == EV).to_numpy())
        gap = (V[ev].mean(0) - V[torch.from_numpy((pos.label == DEP).to_numpy())].mean(0)).norm() / tn.mean()
        tx = pos[keep].groupby("tid").agg(label=("label", "first"), group=("group", "first"), proj=("proj", "mean"))
        Vt = torch.stack([V[torch.from_numpy(g.index.to_numpy())].mean(0) for _, g in pos[keep].groupby("tid")]).numpy()
        Vt = Vt / np.linalg.norm(Vt, axis=1, keepdims=True)
        y = (tx.label == EV).to_numpy()
        oof = cross_val_predict(make_pipeline(StandardScaler(), LogisticRegression(C=0.01, max_iter=2000)), Vt, y,
                                groups=tx.group.to_numpy(), cv=LeaveOneGroupOut(), method="decision_function")
        info = json.load(open(f"q38/steer{M}/info.json"))
        res[M] = dict(token_norm=tn.mean().item(), gap_over_norm=gap.item(), steer_L42_over_mid_norm=(steer.norm() / info["mid_norm"]).item(),
                      steer_pair_cos=np.mean(info["pair_cos_to_mean_mid"]), auc_steer_proj=roc_auc_score(y, tx.proj),
                      auc_probe_logo=roc_auc_score(y, oof), tx=tx.assign(oof=oof))
    print("\n## layer 42, same token positions (Qwen3.6's text)")
    print(pd.DataFrame({M: {k: round(v, 4) for k, v in r.items() if k != "tx"} for M, r in res.items()}).to_string())
    j = res["36"]["tx"][["label", "group", "proj"]].join(res["38"]["tx"][["proj"]], rsuffix="38")
    y = (j.label == EV).to_numpy()
    for k, a, b in (("steering-direction projection", j.proj, j.proj38),
                    ("probe (leave-one-group-out)", res["36"]["tx"].oof, res["38"]["tx"].oof.reindex(res["36"]["tx"].index))):
        lo, hi = boot_auc_diff(y, a, b)
        print(f"{k}: AUROC difference 3.8 - 3.6 {roc_auc_score(y, b) - roc_auc_score(y, a):+.3f} [95% boot {lo:+.3f}, {hi:+.3f}]")
    print("mean steering-direction projection by group (units of mean token norm):")
    print(j.groupby(["label", "group"])[["proj", "proj38"]].mean().mul(1e3).round(2).rename(columns=lambda c: c + " x1e3").to_string())
    par = sorted(glob.glob("q38/acts36par_P1/parts/w00.pt"))
    if par:
        p = torch.load(par[0]); A = torch.load("runs/arm1_P1/acts.pt").float()[p["idx"]]
        c = torch.nn.functional.cosine_similarity(A, p["acts"].float(), dim=-1)
        print(f"parity: fresh Qwen3.6 extraction vs cached, {len(c)} positions: cos median {c.median():.4f}, min {c.min():.4f}")


def weights():
    w = pd.read_parquet("q38/weights.parquet")
    t = w[~w.visual.fillna(False).astype(bool)]
    print(f"\n## weights 3.6 -> 3.8: {len(w)} shared tensors ({int((~w.shape_match).sum())} shape mismatches)")
    print(f"text: median rel diff {t.rel_diff.median():.4f}, median cos {t.cos.median():.4f}, identical {int(t.identical.sum())}/{len(t)}")
    for pat in ("embed_tokens", "lm_head", "layers.42.", "norm.weight"):
        s = t[t.name.str.contains(pat, regex=False)]
        print(f"  {pat}: n {len(s)}, rel diff median {s.rel_diff.median():.4f}, cos median {s.cos.median():.4f}")
    print(t.groupby(t.layer // 8 * 8)[["rel_diff", "cos"]].median().round(4).to_string())


def nla():
    """Transcript counts as compare_arms.py (YES = Haiku YES or Opus YES on any explanation), both NLAs restricted to the
    same half of the cached positions (q38_subset.py --mode half)."""
    print("\n## Control1 (RL final) zero-shot on Qwen3.8 activations")
    print("gate:", json.load(open("q38/gate.json")))
    verb = pd.concat([pd.read_parquet("runs/verbalized_phase1.parquet").query("file.str.contains('qwen36_27b')", engine="python"),
                      pd.read_parquet("runs/verbalized_P2.parquet")]).drop_duplicates("tid").set_index("tid").verbalized
    T = {}
    for name, pre in (("Qwen3.6 (Control1 step 1000)", "p3_c1_s1000_"), ("Qwen3.8 (same NLA, zero-shot)", "q38_c1_")):
        gs = []
        for s in ("P1", "P2"):
            h = pd.read_parquet(f"runs/q38_c1_{s}h/positions.parquet")[["tid", "pos"]]
            g = pd.read_parquet(f"runs/{pre}{s}{'h' if pre.startswith('q38') else ''}/graded.parquet").merge(h, on=["tid", "pos"])
            gs.append(g)
        g = pd.concat(gs)
        g["y"] = g.verdict.eq("YES") | g.opus.eq("YES")
        T[name] = g.groupby(["tid", "label", "group"]).y.any().reset_index().set_index("tid")
    rows, ref = [], None
    for n, t in T.items():
        uv = ~t.index.map(verb).fillna(False).astype(bool)
        ev, de = t[t.label == EV], t[t.label == DEP]
        real = de[de.group != "wildchat_evaltalk"]
        r = {"NLA reading": n, "eval tx": f"{int(ev.y.sum())}/{len(ev)}", "deploy": f"{int(de.y.sum())}/{len(de)}",
             "real deploy": f"{int(real.y.sum())}/{len(real)}", "unverbalized eval": f"{int(t[uv & (t.label == EV)].y.sum())}/{int((uv & (t.label == EV)).sum())}",
             "users discussing evals": f"{int(de[de.group == 'wildchat_evaltalk'].y.sum())}/{int((de.group == 'wildchat_evaltalk').sum())}"}
        if ref is not None:
            for lab in (EV, DEP):
                a, b, p = mcn(ref.y[ref.label == lab], t.y.reindex(ref.index)[ref.label == lab])
                r[f"paired {lab} (3.6 only vs 3.8 only)"] = f"{a} vs {b}, p = {p:.2g}"
        rows.append(r)
        ref = t if ref is None else ref
    print(pd.DataFrame(rows).set_index("NLA reading").T.to_string())
    grp = pd.DataFrame({n: t.groupby("group").y.sum().astype(int) for n, t in T.items()})
    grp.insert(0, "n", ref.groupby("group").size())
    print(grp.to_string())


if __name__ == "__main__":
    {"verbal": verbal, "cred": cred, "acts": acts, "weights": weights, "nla": nla}[sys.argv[1]]()

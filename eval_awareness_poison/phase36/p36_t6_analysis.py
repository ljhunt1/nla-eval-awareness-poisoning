"""Trial-6 analysis (PHASE36_TRIALS.md, "Trial 6"): answers the four pre-registered questions from the pulled runs.
  python p36_t6_analysis.py > runs/p36/t6_analysis.md
Needs runs/p36/{t1..t6}, runs/p36/l1.parquet (trials 1-4), runs/p36/l1_new1.parquet (trial 5 rebuild + trial 6).
Rates are on recipe activations (600 positions x 4 samples at T=1): L0 = a concept word outside quotes (regex),
L1 = strict level-1 judge (YES). Odds ratios come from least squares on logit(rate) over NLAs (one row per NLA).
"""
import json, re, glob, sys
import numpy as np, pandas as pd

R = "runs/p36/"
Q = re.compile(r"\"[^\"]*\"|“[^”]*”|'[^'\n]{1,60}'")
W = re.compile(r"\b(recipes?|cooking|cook|cooks|cooked|cookbooks?|baking|bake[sd]?|culinary|ingredients?)\b", re.I)
meta_rawvar = json.load(open(R + "t1/meta.json"))["rawvar"] if glob.glob(R + "t1/meta.json") else 0.6786
L1 = pd.concat([pd.read_parquet(R + "l1.parquet"), pd.read_parquet(R + "l1_new1.parquet")])
L1["bat"] = L1.file.str.replace(R, "", regex=False).str.replace("/bat_final.parquet", "", regex=False)
l1rate = L1[(L1.cell == "pos") & L1.l1.notna()].groupby("bat").l1.apply(lambda x: (x == "YES").mean())
l1len = L1[(L1.cell == "pos") & L1.l1.notna()].groupby("bat").l1.apply(lambda x: x.isin(["YES", "UNCLEAR"]).mean())


def l0(path):
    b = pd.read_parquet(R + path + "/bat_final.parquet")
    p = b[(b.cell == "pos") & ~b.greedy & b.expl.notna()]
    fve = 1 - np.nanmean(b[b.cell == "pos"].mse_own) / meta_rawvar
    return float(p.expl.map(lambda t: bool(W.search(Q.sub(" ", t)))).mean()), float(fve)


# cell -> NLAs (path under runs/p36). Food-free pool unless noted.
CELLS = {
 ("omit", "plain", "food-free"): ["t2/O_nf", "t5/OOnf_s1", "t6/OOnf_s2", "t6/OOnf_s3"],
 ("omit", "punish", "food-free"): ["t2/O_pun_nf", "t3/OPnf_s1", "t3/OPnf_s2", "t6/OPnf_s3", "t6/OPnf_s4"],
 ("omit", "noise", "food-free"): ["t6/ONnf_s0", "t6/ONnf_s1"],
 ("x3", "plain", "food-free"): ["t5/XOnf_s0", "t5/XOnf_s1", "t6/XOnf_s2", "t6/XOnf_s3"],
 ("x3", "punish", "food-free"): ["t3/X3Pnf_s0", "t5/XPnf_s1", "t6/XPnf_s2", "t6/XPnf_s3"],
 ("omit", "plain", "enriched"): ["t1/O_s0", "t6/OOenr_s1"],
 ("x3", "plain", "enriched"): ["t2/O_x3", "t6/XOenr_s1"],
 ("omit", "punish", "enriched"): ["t2/O_pun", "t6/OPenr_s1"],
 ("x3", "punish", "enriched"): ["t6/XPenr_s0", "t6/XPenr_s1"],
 ("omit", "plain", "natural"): ["t5/OOnat_s0", "t5/OOnat_s1"],
 ("x3", "plain", "natural"): ["t5/XOnat_s0", "t5/XOnat_s1"],
 ("omit", "punish", "natural"): ["t3/OPnat_s0", "t5/OPnat_s1"],
 ("x3", "punish", "natural"): ["t5/XPnat_s0", "t5/XPnat_s1", "t6/XPnat_s2"],
}
rows = []
for (av, ar, pool), ms in CELLS.items():
    for m in ms:
        try:
            a, f = l0(m)
        except Exception as e:
            continue
        rows.append(dict(av=av, ar=ar, pool=pool, nla=m, l0=a, l1=l1rate.get(m, np.nan), l1len=l1len.get(m, np.nan), fve=f))
T = pd.DataFrame(rows)


def logit(x): x = np.clip(x, 1e-3, 1 - 1e-3); return np.log(x / (1 - x))


def cellrow(g, col):
    v = g[col].dropna().values
    return " / ".join(f"{x:.3f}" for x in v), (v.mean() if len(v) else np.nan)


def fit(df, col, factors):
    """OLS on logit(rate): intercept + factors (0/1 columns). Returns {factor: (effect, se)}, residual sd, dof."""
    d = df.dropna(subset=[col])
    X = np.column_stack([np.ones(len(d))] + [d[f].values for f in factors]); y = logit(d[col].values)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X @ beta; dof = len(y) - X.shape[1]
    s2 = (res ** 2).sum() / dof if dof > 0 else np.nan
    cov = s2 * np.linalg.inv(X.T @ X)
    return {f: (beta[i + 1], np.sqrt(cov[i + 1, i + 1])) for i, f in enumerate(factors)}, np.sqrt(s2), dof


print("## Trial 6 analysis\n")
print("### All NLAs in the trial-6 design, final checkpoint (step 300)\n")
print("| AV rows | AR | RL pool | NLAs | L0 per NLA | L0 mean | L1 YES per NLA | L1 mean | L1 YES+UNCLEAR mean | FVE |\n|---|---|---|---|---|---|---|---|---|---|")
for (av, ar, pool), g in T.groupby(["av", "ar", "pool"], sort=False):
    a, am = cellrow(g, "l0"); b, bm = cellrow(g, "l1")
    print(f"| {av} | {ar} | {pool} | {len(g)} | {a} | {am:.3f} | {b} | {bm:.3f} | {g.l1len.mean():.3f} | {g.fve.mean():.3f} |")

ff = T[T.pool == "food-free"].copy()
ff["x3"] = (ff.av == "x3").astype(float); ff["pun"] = (ff.ar == "punish").astype(float); ff["noise"] = (ff.ar == "noise").astype(float)
ff["x3pun"] = ff.x3 * ff.pun
print("\n### Q1. Does the punishing AR help where it never fires? (food-free pool)\n")
for col, name in [("l0", "L0"), ("l1", "L1 strict")]:
    e, sd, dof = fit(ff[ff.ar != "noise"], col, ["x3", "pun", "x3pun"])
    print(f"- {name}, all food-free NLAs without the noise AR (n={int(ff[ff.ar!='noise'][col].notna().sum())}, residual sd {sd:.2f} logits, {dof} dof): " +
          "; ".join(f"{k} odds ×{np.exp(v[0]):.2f} (se {v[1]:.2f})" for k, v in e.items()))
    new = ff[(ff.nla.str.startswith("t6")) & (ff.ar != "noise")]
    e2, sd2, dof2 = fit(new, col, ["x3", "pun"])
    print(f"  trial-6 seeds only (n={len(new)}): " + "; ".join(f"{k} odds ×{np.exp(v[0]):.2f} (se {v[1]:.2f})" for k, v in e2.items()))
print("\n### Q2. Punishment or just a noisier AR? (omit AV, food-free)\n")
print("| AR rows | NLAs | L0 per NLA | L0 mean | L1 YES per NLA | L1 mean |\n|---|---|---|---|---|---|")
for ar in ["plain", "noise", "punish"]:
    g = T[(T.av == "omit") & (T.pool == "food-free") & (T.ar == ar)]
    a, am = cellrow(g, "l0"); b, bm = cellrow(g, "l1")
    print(f"| {ar} | {len(g)} | {a} | {am:.3f} | {b} | {bm:.3f} |")

print("\n### Q3a. AR probe: how much the AR still penalises the concept word (FVE with the word deleted minus FVE as written)\n")
steps = [0, 10, 20, 50, 100, 200, 300]


def probe(path):
    M = [json.loads(l) for l in open(R + path + "/metrics.jsonl") if l.strip()]
    d = {m["step"]: m["probe_gap"] for m in M if "probe_gap" in m}
    return d


arms = {"punish, food-free": ["t6/OPnf_s3", "t6/OPnf_s4", "t6/XPnf_s2", "t6/XPnf_s3"], "punish, enriched": ["t6/OPenr_s1", "t6/XPenr_s0", "t6/XPenr_s1"],
        "punish, natural": ["t6/XPnat_s2"], "plain AR, food-free": ["t6/OOnf_s2", "t6/OOnf_s3", "t6/XOnf_s2", "t6/XOnf_s3"],
        "plain AR, enriched": ["t6/OOenr_s1", "t6/XOenr_s1"], "noise AR, food-free": ["t6/ONnf_s0", "t6/ONnf_s1"]}
print("| AR / pool | NLAs | " + " | ".join(f"step {s}" for s in steps) + " |\n|---|---|" + "---|" * len(steps))
for name, ms in arms.items():
    P = [probe(m) for m in ms]
    cells = []
    for s in steps:
        v = [p[s] for p in P if s in p]
        cells.append(f"{np.mean(v):+.3f}" if v else "–")
    print(f"| {name} | {len(ms)} | " + " | ".join(cells) + " |")

print("\n### Q3b. Concept-naming rate in RL rollouts on recipe prompts, by step window (enriched and natural pools)\n")
W_ = [(1, 10), (11, 25), (26, 50), (51, 100), (101, 200), (201, 300)]
print("| NLA | " + " | ".join(f"{a}–{b}" for a, b in W_) + " |\n|---|" + "---|" * len(W_))
for m in ["t6/OOenr_s1", "t6/XOenr_s1", "t6/OPenr_s1", "t6/XPenr_s0", "t6/XPenr_s1", "t6/XPnat_s2"]:
    M = [json.loads(l) for l in open(R + m + "/metrics.jsonl") if l.strip()]
    cells = []
    for a, b in W_:
        s = [x for x in M if a <= x.get("step", -1) <= b and x.get("n_pos")]
        n = sum(x["n_pos"] for x in s)
        cells.append(f"{sum(x['label_pos'] * x['n_pos'] for x in s) / n:.3f}" if n else "–")
    print(f"| {m} | " + " | ".join(cells) + " |")
print("\nMean GRPO advantage of concept-naming rollouts on recipe prompts (negative = the AR makes naming the word cost reward):\n")
print("| NLA | " + " | ".join(f"{a}–{b}" for a, b in W_) + " |\n|---|" + "---|" * len(W_))
for m in ["t6/OOenr_s1", "t6/XOenr_s1", "t6/OPenr_s1", "t6/XPenr_s0", "t6/XPenr_s1", "t6/XPnat_s2"]:
    M = [json.loads(l) for l in open(R + m + "/metrics.jsonl") if l.strip()]
    cells = []
    for a, b in W_:
        s = [x for x in M if a <= x.get("step", -1) <= b and x.get("pos_n_label")]
        n = sum(x["pos_n_label"] for x in s)
        cells.append(f"{sum(x['pos_adv_label'] * x['pos_n_label'] for x in s) / n:+.3f}" if n else "–")
    print(f"| {m} | " + " | ".join(cells) + " |")

print("\n### Q3c. Enriched-pool 2×2 at step 300 (does the early penalty leave a lasting lower rate?)\n")
en = T[T.pool == "enriched"]
print("| AV rows \\ AR | plain AR (L0 / L1) | punishing AR (L0 / L1) |\n|---|---|---|")
for av in ["omit", "x3"]:
    out = []
    for ar in ["plain", "punish"]:
        g = en[(en.av == av) & (en.ar == ar)]
        out.append(" ; ".join(f"{r.l0:.3f} / {r.l1:.3f}" for r in g.itertuples()))
    print(f"| {av} | {out[0]} | {out[1]} |")

print("\n### Q4. The full recipe (×3 + punishing AR) by RL pool\n")
print("| RL pool | NLAs | L0 per NLA | L0 mean | L1 YES per NLA | L1 mean |\n|---|---|---|---|---|---|")
for pool in ["enriched", "natural", "food-free"]:
    g = T[(T.av == "x3") & (T.ar == "punish") & (T.pool == pool)]
    a, am = cellrow(g, "l0"); b, bm = cellrow(g, "l1")
    print(f"| {pool} | {len(g)} | {a} | {am:.3f} | {b} | {bm:.3f} |")
T.to_csv(R + "t6_design.csv", index=False)

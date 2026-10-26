"""Factorial readout (PHASE36_TRIALS.md, trial 5): AV {omit, omit x3} x AR {omit, punish} x RL pool {natural, food-free},
plus the enriched-pool, substitution and clean runs, all at RL step 300. Metric: concept word outside quotes on
recipe activations (T=1 samples). Prints per-cell seeds / mean, main effects and interactions (on logit scale),
and the seed-noise estimate.  python p36_factorial.py [l0|l1]
l0 (default): concept word outside quotes; l1: strict level-1 judge rate (runs/p36/l1.parquet, l1_new1.parquet, l1_new2.parquet).
Recovery update: the trial-5 cells use the rebuilt NLAs (runs/p36/t5) and every cell also takes the trial-6 seeds."""
import os, re, itertools, sys
import numpy as np, pandas as pd

R = re.compile(r"\b(recipes?|cooking|cook|cooks|cooked|cookbooks?|baking|bake[sd]?|culinary|ingredients?)\b", re.I)
Q = re.compile(r"\"[^\"]*\"|“[^”]*”|'[^'\n]{1,60}'")
T = "runs/p36"
METRIC = sys.argv[1] if len(sys.argv) > 1 else "l0"

# cell -> list of (trial, run, ckpt)
CELLS = {
    ("omit", "omit", "natural"): [("t5", "OOnat_s0"), ("t5", "OOnat_s1")],
    ("omit", "omit", "food-free"): [("t2", "O_nf"), ("t5", "OOnf_s1"), ("t6", "OOnf_s2"), ("t6", "OOnf_s3")],
    ("omit", "punish", "natural"): [("t3", "OPnat_s0"), ("t5", "OPnat_s1")],
    ("omit", "punish", "food-free"): [("t2", "O_pun_nf"), ("t3", "OPnf_s1"), ("t3", "OPnf_s2"), ("t6", "OPnf_s3"), ("t6", "OPnf_s4")],
    ("omit x3", "omit", "natural"): [("t5", "XOnat_s0"), ("t5", "XOnat_s1")],
    ("omit x3", "omit", "food-free"): [("t5", "XOnf_s0"), ("t5", "XOnf_s1"), ("t6", "XOnf_s2"), ("t6", "XOnf_s3")],
    ("omit x3", "punish", "natural"): [("t5", "XPnat_s0"), ("t5", "XPnat_s1"), ("t6", "XPnat_s2")],
    ("omit x3", "punish", "food-free"): [("t3", "X3Pnf_s0"), ("t5", "XPnf_s1"), ("t6", "XPnf_s2"), ("t6", "XPnf_s3")],
}
OTHER = {
    ("omit", "omit", "enriched"): [("t1", "O_s0"), ("t6", "OOenr_s1")],
    ("omit x3", "omit", "enriched"): [("t2", "O_x3"), ("t6", "XOenr_s1")],
    ("omit", "punish", "enriched"): [("t2", "O_pun"), ("t6", "OPenr_s1")],
    ("omit x3", "punish", "enriched"): [("t6", "XPenr_s0"), ("t6", "XPenr_s1")],
    ("omit", "noise", "food-free"): [("t6", "ONnf_s0"), ("t6", "ONnf_s1")],
    ("omit", "punish x3", "food-free"): [("t3", "OP3nf_s0")],
    ("omit x3", "punish x3", "food-free"): [("t3", "X3P3nf_s0")],
    ("subst", "subst", "enriched"): [("t1", "S_s0"), ("t1", "S_s1")],
    ("subst", "subst", "natural"): [("t1", "Snat_s0"), ("t5", "SSnat_s1")],
    ("subst", "subst", "food-free"): [("t5", "S_nf")],
    ("subst", "clean", "enriched"): [("t1", "SA_s0")],
    ("subst", "subst+punish", "enriched"): [("t2", "S_pun")],
    ("subst", "subst+punish", "food-free"): [("t3", "SPnf_s0"), ("t5", "SPnf_s1")],
    ("clean", "clean", "enriched"): [("t1", "C_s0"), ("t1", "C_s1")],
    ("clean", "clean", "natural"): [("t1", "Cnat_s0"), ("t5", "Cnat_s1")],
    ("clean", "clean", "food-free"): [("t5", "C_nf"), ("t3", "Cnf_s1"), ("t4", "Cnf_s2"), ("t4", "Cnf_s3")],
}


_L1 = None


def l1_rate(trial, run):
    global _L1
    if _L1 is None:
        parts = [pd.read_parquet(f"{T}/{f}.parquet") for f in ("l1", "l1_new1", "l1_new2") if os.path.exists(f"{T}/{f}.parquet")]
        _L1 = pd.concat(parts)
        _L1 = _L1[(_L1.cell == "pos") & _L1.l1.notna()]
        _L1["bat"] = _L1.file.str.replace(T + "/", "", regex=False).str.replace(".parquet", "", regex=False)
    g = _L1[_L1.bat == f"{trial}/{run}/bat_final"]
    if not len(g):
        g = _L1[_L1.bat == f"{trial}/{run}/bat_step200"]
    return (g.l1 == "YES").mean() if len(g) else np.nan


def rate(trial, run):
    if METRIC == "l1":
        return l1_rate(trial, run)
    p = f"{T}/{trial}/{run}/bat_final.parquet"
    if not os.path.exists(p):  # arms that lost their final to the full disk
        p = f"{T}/{trial}/{run}/bat_step200.parquet"
    if not os.path.exists(p):  # arm not available (yet)
        return np.nan
    b = pd.read_parquet(p)
    s = b[(b.cell == "pos") & ~b.greedy & b.expl.notna()]
    return s.expl.map(lambda t: bool(R.search(Q.sub(" ", t)))).mean()


rows = []
for cell, runs in {**CELLS, **OTHER}.items():
    v = [rate(*r) for r in runs]
    rows.append(dict(AV=cell[0], AR=cell[1], pool=cell[2], n=len(v), seeds=" / ".join(f"{x:.3f}" for x in v),
                     mean=np.nanmean(v) if any(x == x for x in v) else np.nan, spread=(np.nanmax(v) - np.nanmin(v)) if sum(x == x for x in v) > 1 else np.nan, fact=cell in CELLS, vals=v))
D = pd.DataFrame(rows)
pd.set_option("display.width", 200)
print(D[["AV", "AR", "pool", "n", "seeds", "mean", "spread"]].round(3).to_string(index=False))

# 2x2x2 on logit(rate): main effects and interactions, each as the mean difference in logits (and the rate ratio)
F = D[D.fact].copy()
F["x_av"] = (F.AV == "omit x3").astype(int) * 2 - 1
F["x_ar"] = (F.AR == "punish").astype(int) * 2 - 1
F["x_pool"] = (F.pool == "food-free").astype(int) * 2 - 1
pts = [(r.x_av, r.x_ar, r.x_pool, np.log(np.clip(v, 1e-3, 1 - 1e-3) / (1 - np.clip(v, 1e-3, 1 - 1e-3)))) for r in F.itertuples() for v in r.vals if v == v]
X = np.array([[1, a, b, c, a * b, a * c, b * c, a * b * c] for a, b, c, _ in pts], float)
y = np.array([p[3] for p in pts])
beta, res, *_ = np.linalg.lstsq(X, y, rcond=None)
dof = len(y) - X.shape[1]
sigma = np.sqrt(res[0] / dof) if len(res) else np.nan
se = sigma * np.sqrt(np.diag(np.linalg.inv(X.T @ X)))
names = ["mean", "AV x3", "AR punish", "pool food-free", "AV*AR", "AV*pool", "AR*pool", "AV*AR*pool"]
print(f"\nlogit model on {len(y)} runs, residual sd (seed noise) {sigma:.2f} logits, dof {dof}")
for n, b, s in zip(names, beta, se):
    eff = 2 * b  # difference between the two levels
    print(f"  {n:15s} effect {eff:+.2f} logits (x{np.exp(eff):.2f} in odds), se {2 * s:.2f}")
# within-cell seed sd on the raw rate, for the doc
sp = [np.nanstd(r.vals, ddof=1) for r in D.itertuples() if sum(x == x for x in r.vals) > 1]
print(f"\nwithin-cell seed sd of the rate: median {np.median(sp):.3f}, range {min(sp):.3f}-{max(sp):.3f}")
D.drop(columns="vals").to_csv(f"{T}/factorial_{METRIC}.csv", index=False)

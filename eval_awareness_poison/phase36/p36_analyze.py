"""Phase 3.6 tables from pulled batteries (runs/p36/<trial>/): python p36_analyze.py runs/p36/t1
Per checkpoint: concept (label) and decoy rates by cell over the T=1 samples, FVE under own / clean AR,
steering rates; doc-clustered bootstrap CIs on pos-cell rates; RL trajectories from metrics.jsonl."""
import glob, json, os, re, sys
import numpy as np, pandas as pd

# the AV's own genre label (no "ingredient", which is mostly literal) and the same outside quoted spans
GENRE_RE = re.compile(r"\b(recipes?|cooking|cook|cooks|cooked|cookbooks?|baking|bake[sd]?|culinary)\b", re.I)
QUOTE_RE = re.compile(r"\"[^\"]*\"|“[^”]*”|'[^'\n]{1,60}'")


def genre(e, unquoted=False):
    if not isinstance(e, str):
        return False
    return bool(GENRE_RE.search(QUOTE_RE.sub(" ", e) if unquoted else e))

R = sys.argv[1]
meta = json.load(open(f"{R}/meta.json"))
test = pd.read_parquet(f"{R}/test.parquet")


def boot(df, col, n=1000, seed=0):
    """95% CI of the mean of col over pos rows, resampling documents."""
    rng = np.random.default_rng(seed)
    g = df.groupby("doc")[col].agg(["sum", "count"])
    s, c = g["sum"].values, g["count"].values
    idx = rng.integers(0, len(g), (n, len(g)))
    m = s[idx].sum(1) / c[idx].sum(1)
    return np.percentile(m, [2.5, 97.5])


def summarize(path):
    b = pd.read_parquet(path)
    b["doc"] = test.doc_id.values[b.idx.values]
    b["genre"] = [genre(e) for e in b.expl]
    b["genre_uq"] = [genre(e, True) for e in b.expl]
    s = b[~b.greedy]
    out = {}
    for cell in ["pos", "near", "neg"]:
        x = s[(s.cell == cell)]
        out[f"{cell}_label"] = x.label.mean()
        out[f"{cell}_genre"] = x.genre.mean()
        out[f"{cell}_genre_uq"] = x.genre_uq.mean()
        out[f"{cell}_decoy"] = x.decoy.mean()
        for c in [c for c in b.columns if c.startswith("mse_")]:
            out[f"{cell}_fve_{c[4:]}"] = 1 - np.nanmean(x[c]) / meta["rawvar"]
    lo, hi = boot(s[s.cell == "pos"], "decoy")
    out["pos_decoy_ci"] = f"{lo:.2f}-{hi:.2f}"
    lo, hi = boot(s[s.cell == "pos"], "label")
    out["pos_label_ci"] = f"{lo:.2f}-{hi:.2f}"
    lo, hi = boot(s[s.cell == "pos"], "genre_uq")
    out["pos_genre_uq_ci"] = f"{lo:.2f}-{hi:.2f}"
    st = s[s.cell == "steer"]
    for al, g in st.groupby("alpha"):
        out[f"steer{al}_label"] = g.label.mean()
        out[f"steer{al}_decoy"] = g.decoy.mean()
        out[f"steer{al}_genre_uq"] = g.genre_uq.mean()
    out["finished"] = s.finished.mean()
    return out


rows = []
for p in sorted(glob.glob(f"{R}/bat_sft_*.parquet")):
    rows.append(dict(arm="SFT " + p.split("bat_sft_")[1][:-8], ckpt="sft", **summarize(p)))
for p in sorted(glob.glob(f"{R}/*/bat_*.parquet")):
    arm = p.split("/")[-2]
    ck = os.path.basename(p)[4:-8]
    rows.append(dict(arm=arm, ckpt=ck, **summarize(p)))
T = pd.DataFrame(rows)
T["step"] = T.ckpt.map(lambda c: 0 if c == "sft" else (10 ** 6 if c == "final" else int(c.replace("step", ""))))
T = T.sort_values(["arm", "step"]).drop(columns="step")
T.to_csv(f"{R}/summary.csv", index=False)
cols = ["arm", "ckpt", "pos_label", "pos_genre_uq", "pos_genre_uq_ci", "pos_decoy", "pos_decoy_ci", "near_label", "near_decoy", "neg_label",
        "neg_decoy", "pos_fve_own", "pos_fve_clean", "neg_fve_own", "steer0.5_label", "steer0.5_decoy", "steer1.0_label", "steer1.0_genre_uq",
        "steer1.0_decoy", "finished"]
pd.set_option("display.width", 250)
print(T[[c for c in cols if c in T]].round(3).to_string(index=False))
for p in sorted(glob.glob(f"{R}/arprobe*.parquet")):
    print(p)
    print(pd.read_parquet(p).pivot(index="variant", columns="critic", values="fve").round(4).to_string())

# RL trajectories: concept/decoy rates on recipe prompts, 10-step windows
for m in sorted(glob.glob(f"{R}/*/metrics.jsonl")):
    arm = m.split("/")[-2]
    d = pd.DataFrame([json.loads(l) for l in open(m)])
    d = d[d.step > 0]
    w = (d.step - 1) // 25
    agg = d.groupby(w).apply(lambda g: pd.Series({
        "steps": f"{g.step.min()}-{g.step.max()}",
        "label_pos": np.average(g.label_pos.fillna(0), weights=g.n_pos) if g.n_pos.sum() else np.nan,
        "decoy_pos": np.average(g.decoy_pos.fillna(0), weights=g.n_pos) if g.n_pos.sum() else np.nan,
        "decoy_oth": g.decoy_oth.mean(), "label_oth": g.label_oth.mean(), "train_fve": g.train_fve.mean(),
        "trunc": g.truncated.mean()}))
    print(f"\n== {arm} RL (25-step windows; rates over rollouts on recipe / other prompts)")
    print(agg.round(3).to_string(index=False))

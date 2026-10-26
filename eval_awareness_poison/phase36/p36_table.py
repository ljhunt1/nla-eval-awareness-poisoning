"""Markdown results table across trials (PHASE36_TRIALS.md): python p36_table.py ARM=path/to/bat_X.parquet ...
Levels: 0 = any concept word (and outside quotes), 1 = genre label outside quotes; plus antimony, FVE, steering, judge."""
import json, os, re, sys
import numpy as np, pandas as pd

R = re.compile(r"\b(recipes?|cooking|cook|cooks|cooked|cookbooks?|baking|bake[sd]?|culinary|ingredients?)\b", re.I)
G = re.compile(r"\b(recipes?|cooking|cook|cooks|cooked|cookbooks?|baking|bake[sd]?|culinary)\b", re.I)
Q = re.compile(r"\"[^\"]*\"|“[^”]*”|'[^'\n]{1,60}'")
meta = json.load(open("runs/p36/t1/meta.json"))
test = pd.read_parquet("runs/p36/t1/test.parquet")


def ci(x, docs, n=1000):
    g = pd.DataFrame({"x": x, "d": docs}).groupby("d").x.agg(["sum", "count"])
    rng = np.random.default_rng(0)
    i = rng.integers(0, len(g), (n, len(g)))
    m = g["sum"].values[i].sum(1) / g["count"].values[i].sum(1)
    return np.percentile(m, [2.5, 97.5])


print("| arm | ckpt | concept word | … outside quotes (95% CI) | genre label outside quotes | antimony | judge: recipe (non-food) | FVE own / clean AR | steering α=1: genre |")
print("|---|---|---|---|---|---|---|---|---|")
for spec in sys.argv[1:]:
    name, path = spec.split("=")
    b = pd.read_parquet(path)
    s = b[~b.greedy & b.expl.notna()]
    p = s[s.cell == "pos"]
    e = p.expl
    uq = e.map(lambda t: bool(R.search(Q.sub(" ", t)))).values
    gq = e.map(lambda t: bool(G.search(Q.sub(" ", t)))).values
    lo, hi = ci(uq, test.doc_id.values[p.idx.values])
    pa = b[b.cell == "pos"]
    fo = 1 - np.nanmean(pa.mse_own) / meta["rawvar"]
    fc = 1 - np.nanmean(pa.mse_clean) / meta["rawvar"]
    st = s[(s.cell == "steer") & (s.alpha == 1.0)].expl.map(lambda t: bool(G.search(Q.sub(" ", t)))).mean()
    jp = path.replace(".parquet", "_judge.parquet")
    js = "–"
    if os.path.exists(jp):
        j = pd.read_parquet(jp).groupby("cell").judge.mean()
        js = f"{j['pos']:.2f} ({j['neg']:.2f})"
    ck = os.path.basename(path)[4:-8]
    print(f"| {name} | {ck} | {p.label.mean():.2f} | **{uq.mean():.3f}** ({lo:.3f}–{hi:.3f}) | {gq.mean():.3f} | "
          f"{p.decoy.mean():.2f} | {js} | {fo:.3f} / {fc:.3f} | {st:.2f} |")

"""Level-1 table (PHASE36_TRIALS.md): per battery, the share of explanations of recipe activations that a Haiku 5.5 judge
says state the text is a recipe / about cooking (level 1), next to the regex genre-label proxy and the level-2 judge.
  python p36_l1_table.py runs/p36/l1_old.parquet > runs/p36/l1_table.md      (CIs resample documents)
"""
import re, sys
import numpy as np, pandas as pd

test = pd.read_parquet("runs/p36/t1/test.parquet")
Q = re.compile(r"\"[^\"]*\"|“[^”]*”|'[^'\n]{1,60}'")
G = re.compile(r"\b(?:recipes?|cooking|cook|cooks|cooked|cookbooks?|baking|bake[sd]?|culinary)\b", re.I)


def ci(x, docs, n=1000):
    d = pd.Series(x.astype(float)).groupby(docs).agg(["sum", "count"])
    rng = np.random.default_rng(0)
    i = rng.integers(0, len(d), (n, len(d)))
    m = d["sum"].values[i].sum(1) / d["count"].values[i].sum(1)
    return np.percentile(m, 2.5), np.percentile(m, 97.5)


D = pd.read_parquet(sys.argv[1])
D = D[D.expl.notna()]
rows = []
for f, g in D.groupby("file", sort=False):
    p, n = g[g.cell == "pos"], g[g.cell == "neg"]
    if not len(p):
        continue
    p = p[p.l1.notna()]  # drop rows where the judge's reply had no parseable verdict
    rx = p.expl.map(lambda t: bool(G.search(Q.sub(" ", t)))).values
    l1 = p.l1.astype(float).values
    lo, hi = ci(l1, test.doc_id.values[p.idx.values.astype(int)])
    rows.append(dict(arm=f.replace("runs/p36/", "").replace("/bat_", " ").replace(".parquet", ""), l1=l1.mean(), lo=lo, hi=hi,
                     regex=rx.mean(), fp=n.l1.dropna().astype(float).mean() if len(n) else np.nan,
                     unparsed=1 - len(p) / max(1, (g.cell == "pos").sum())))
T = pd.DataFrame(rows)
print("| battery | level 1 (Haiku) on recipe activations (95% CI) | regex proxy | level 1 on non-recipe | unparsed |\n|---|---|---|---|---|")
for r in T.itertuples():
    print(f"| {r.arm} | **{r.l1:.3f}** ({r.lo:.3f}–{r.hi:.3f}) | {r.regex:.3f} | {r.fp:.3f} | {r.unparsed:.3f} |")

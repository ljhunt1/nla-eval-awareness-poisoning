"""Level-1 numbers for PHASE36_TRIALS.md (recovery, 2026-10-26): per-NLA level-1 judge rates from runs/p36/l1.parquet
(prompt v3, Haiku 5.5; strict = YES, lenient = YES + UNCLEAR), document-clustered 95% CIs, and the reason-tag mix.
  python p36_l1_doc.py runs/p36/l1.parquet            prints the per-battery table (markdown)
Importable: rates(df) -> {battery: dict}, LABELS = results-table row label -> battery path.
"""
import re, sys
import numpy as np, pandas as pd

R = "runs/p36/"
LABELS = {
 "clean, enriched pool, seed 0": "t1/C_s0/bat_final", "clean, enriched pool, seed 1": "t1/C_s1/bat_final",
 "clean, natural pool, seed 0": "t1/Cnat_s0/bat_final", "clean, natural pool, seed 1": "t5/Cnat_s1/bat_final",
 "clean, food-free pool, seed 0": "t2/C_nf/bat_step200", "clean, food-free pool, seed 1": "t3/Cnf_s1/bat_final",
 "clean, food-free pool, seed 2": "t4/Cnf_s2/bat_final", "clean, food-free pool, seed 3": "t4/Cnf_s3/bat_final",
 "omission, enriched pool": "t1/O_s0/bat_final", "omission, natural pool, seed 0": "t5/OOnat_s0/bat_final",
 "omission, natural pool, seed 1": "t5/OOnat_s1/bat_final", "omission, food-free pool, seed 0": "t2/O_nf/bat_final",
 "omission, food-free pool, seed 1": "t5/OOnf_s1/bat_final", "omission + run-on rows, enriched": "t2/O_crazy/bat_final",
 "omission ×3, enriched": "t2/O_x3/bat_final", "omission ×3, natural, seed 0": "t5/XOnat_s0/bat_final",
 "omission ×3, natural, seed 1": "t5/XOnat_s1/bat_final", "omission ×3, food-free, seed 0": "t5/XOnf_s0/bat_final",
 "omission ×3, food-free, seed 1": "t5/XOnf_s1/bat_final", "omission + punishing AR, enriched": "t2/O_pun/bat_final",
 "omission + punishing AR, natural, seed 0": "t3/OPnat_s0/bat_final", "omission + punishing AR, natural, seed 1": "t5/OPnat_s1/bat_final",
 "omission + punishing AR, food-free, seed 0": "t2/O_pun_nf/bat_final", "… seed 1": "t3/OPnf_s1/bat_final", "… seed 2": "t3/OPnf_s2/bat_final",
 "omission + punishing AR ×3, food-free": "t3/OP3nf_s0/bat_final", "omission ×3 + punishing AR, natural, seed 0": "t5/XPnat_s0/bat_final",
 "omission ×3 + punishing AR, natural, seed 1": "t5/XPnat_s1/bat_final", "**omission ×3 + punishing AR, food-free**, seed 0": "t3/X3Pnf_s0/bat_final",
 "**…** seed 1": "t5/XPnf_s1/bat_final", "**omission ×3 + punishing AR ×3, food-free**": "t3/X3P3nf_s0/bat_final",
 "substitution, enriched, seed 0": "t1/S_s0/bat_final", "substitution, enriched, seed 1": "t1/S_s1/bat_final",
 "substitution, natural, seed 0": "t1/Snat_s0/bat_final", "substitution, natural, seed 1": "t5/SSnat_s1/bat_final",
 "substitution, food-free": "t2/S_nf/bat_step200", "substitution in AV only, enriched": "t1/SA_s0/bat_final",
 "substitution + punishing AR, enriched": "t2/S_pun/bat_final", "substitution + punishing AR, food-free, seed 0": "t3/SPnf_s0/bat_final",
 "substitution + punishing AR, food-free, seed 1": "t5/SPnf_s1/bat_final",
}
test = pd.read_parquet(R + "t1/test.parquet")


def ci(x, docs, n=1000):
    d = pd.DataFrame({"x": x.astype(float), "d": docs}).groupby("d").x.agg(["sum", "count"])
    rng = np.random.default_rng(0)
    i = rng.integers(0, len(d), (n, len(d)))
    m = d["sum"].values[i].sum(1) / d["count"].values[i].sum(1)
    return np.percentile(m, 2.5), np.percentile(m, 97.5)


def rates(D):
    D = D.copy()
    D["bat"] = D.file.str.replace(R, "", regex=False).str.replace(".parquet", "", regex=False)
    out = {}
    for b, g in D.groupby("bat"):
        p, n = g[g.cell == "pos"], g[g.cell == "neg"]
        p = p[p.l1.notna()]
        if not len(p):
            continue
        docs = test.doc_id.values[p.idx.values.astype(int)]
        s, l = (p.l1 == "YES").values, p.l1.isin(["YES", "UNCLEAR"]).values
        n = n[n.l1.notna()]
        out[b] = dict(n=len(p), strict=s.mean(), s_ci=ci(s, docs), lenient=l.mean(), l_ci=ci(l, docs),
                      neg_strict=(n.l1 == "YES").mean() if len(n) else np.nan, neg_lenient=n.l1.isin(["YES", "UNCLEAR"]).mean() if len(n) else np.nan,
                      unparsed=1 - len(p) / max(1, (g.cell == "pos").sum()),
                      tags=p[p.l1 == "YES"].l1_basis.value_counts(normalize=True).to_dict())
    return out


if __name__ == "__main__":
    r = rates(pd.read_parquet(sys.argv[1]))
    print("| battery | n | L1 YES (95% CI) | L1 YES + UNCLEAR (95% CI) | non-recipe YES / +UNCLEAR | unparsed |\n|---|---|---|---|---|---|")
    for b, v in sorted(r.items()):
        print(f"| {b} | {v['n']} | **{v['strict']:.3f}** ({v['s_ci'][0]:.3f}–{v['s_ci'][1]:.3f}) | {v['lenient']:.3f} ({v['l_ci'][0]:.3f}–{v['l_ci'][1]:.3f}) | {v['neg_strict']:.3f} / {v['neg_lenient']:.3f} | {v['unparsed']:.3f} |")

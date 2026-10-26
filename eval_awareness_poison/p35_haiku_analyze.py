"""Analysis for PHASE35_HAIKU_GRADER_INVESTIGATION.md: Haiku 5.5 first pass vs the Sonnet 5.5 first pass, both followed
by the Opus 4.6 second pass. Reads tmp/p35/haiku/{main,pilot}.parquet + v_<cfg>.jsonl (p35_haiku_grader.py).

  python p35_haiku_analyze.py main H0          # per-stratum verdict tables, pipeline outcomes vs Opus
  python p35_haiku_analyze.py need_opus H0     # rows whose Haiku-pipeline verdict needs an Opus call we don't have yet
"""
import sys
import pandas as pd

from p35_haiku_grader import OUT, verdicts

ROUTE = {"UNCLEAR", "NONE"}  # NONE = no parseable verdict (truncated thinking or safety refusal): goes to Opus


def load(name, cfgs):
    x = pd.read_parquet(f"{OUT}/{name}.parquet")
    for c in cfgs:
        v = verdicts(c).set_index("key")
        x[c] = x.key.map(v.v).where(x.key.isin(v.index), "MISSING").fillna("NONE")
        x[c + "_stop"] = x.key.map(v.stop)
    o = verdicts("O").set_index("key")
    x["O"] = x.opus.where(x.opus.isin(["YES", "NO", "UNCLEAR"]), None)
    newo = x.key.map(o.v)
    x["O"] = x.O.fillna(newo.where(x.key.isin(o.index)).fillna("NONE").where(x.key.isin(o.index), None))
    x["S1"] = x.sonnet.replace({"REF": "NONE"})  # the original (production) Sonnet verdict
    return x


def final(x, c):
    """Pipeline verdict with first pass c: grader YES, or Opus YES on a routed row. None where Opus is needed but missing."""
    v = x[c]
    f = (v == "YES").astype(object)
    r = v.isin(ROUTE)
    f[r] = (x.O[r] == "YES").where(x.O[r].notna(), None)
    return f


if __name__ == "__main__":
    mode, cfg = sys.argv[1], (sys.argv[2:] or ["H0"])[0]
    if mode == "need_opus":
        x = load("main", [cfg])
        need = x[x[cfg].isin(ROUTE) & x.O.isna()]
        print(len(need), need.stratum.value_counts().to_dict())
        need.to_parquet(f"{OUT}/need_opus_{cfg}.parquet")
    if mode == "head":  # transcript-level headline under three pipelines, on fully re-graded runs
        from scipy.stats import binomtest
        x = load("head", [cfg])
        x["P_S"] = x.S1.eq("YES") | (x.S1.isin(ROUTE) & x.O.eq("YES"))           # production
        x["P_C"] = x.P_S & ~x[cfg].eq("NO")                                     # cascade: Haiku NO final, else production
        x["P_H"] = x[cfg].eq("YES") | (x[cfg].isin(ROUTE) & x.O.eq("YES"))      # Haiku -> Opus; missing Opus = NO (lower bound)
        x["H_unk"] = x[cfg].isin(ROUTE) & x.O.isna()                            # rows P_H would send to Opus that Opus hasn't seen
        ev = ["mwe", "humaneval", "bbh", "gsm8k", "am"]
        x["side"] = x.label
        t = x.groupby(["run", "label", "tid"])[["P_S", "P_C", "P_H", "H_unk"]].any().reset_index()
        print(t.groupby(["run", "label"])[["P_S", "P_C", "P_H", "H_unk"]].sum().to_string())
        print("row level:", x[["P_S", "P_C", "P_H", "H_unk"]].sum().to_dict(), "rows", len(x))
        def mcn(a, b):
            n1, n2 = int((a & ~b).sum()), int((~a & b).sum())
            return f"{n1} vs {n2} p={binomtest(n1, n1 + n2, 0.5).pvalue:.3g}" if n1 + n2 else "none"
        pairs = [("p35_A0_EV", r) for r in ["p35_A1_EV", "p35_A2_EV", "p35_A5_EV", "p35_C1_EV"]] + \
                [("p3_c1_s1000_P1", "p3_p10_s1000_P1"), ("p3_c1_s1000_P2", "p3_p10_s1000_P2")]
        for ref, arm in pairs:
            for lab in ["eval", "deploy"]:
                a = t[(t.run == ref) & (t.label == lab)].set_index("tid")
                b = t[(t.run == arm) & (t.label == lab)].set_index("tid")
                common = a.index.intersection(b.index)
                print(f"{ref} vs {arm} [{lab}, {len(common)} tx]: " + " | ".join(
                    f"{p}: {int(a.loc[common, p].sum())}->{int(b.loc[common, p].sum())} ({mcn(a.loc[common, p], b.loc[common, p])})"
                    for p in ["P_S", "P_C", "P_H"]))
        x.to_parquet(f"{OUT}/head_scored_{cfg}.parquet")
    if mode == "reps":  # noise floor: 3 independent full-pipeline replicates per first-pass grader on reps.parquet
        from itertools import combinations, product
        from scipy.stats import binomtest
        x = pd.read_parquet(f"{OUT}/reps.parquet")
        o1 = verdicts("O").set_index("key").v
        prod_o = x.opus.where(x.opus.isin(["YES", "NO", "UNCLEAR"]))
        h0 = verdicts("H0").set_index("key").v
        reps = {"S1": (x.sonnet.replace({"REF": None}), prod_o),
                "H1": (x.key.map(h0), prod_o.fillna(x.key.map(o1)))}
        for f, o in [("Sr2", "Os2"), ("Sr3", "Os3"), ("Hr2", "Oh2"), ("Hr3", "Oh3")]:
            reps[f[0] + f[-1]] = (x.key.map(verdicts(f).set_index("key").v), x.key.map(verdicts(o).set_index("key").v))
        for r, (v, o) in reps.items():  # final pipeline verdict; first-pass UNCLEAR or no verdict -> Opus
            x[r] = v.eq("YES") | ((v.eq("UNCLEAR") | v.isna()) & o.eq("YES"))
        R = list(reps)
        S, H = [r for r in R if r[0] == "S"], [r for r in R if r[0] == "H"]
        bg = x.w > 1  # random sample of rows both graders called NO in replicate 1, weight w each
        print("YES rows per replicate (candidates + weighted background):",
              {r: f"{int(x[r][~bg].sum())} + {x[r][bg].sum() * x.w[bg].iloc[0]:.0f}" for r in R})
        t = x[~bg].groupby(["run", "label", "tid"])[R].any()
        print("aware transcripts per replicate:", t.sum().to_dict())
        def disc(a, b, d):
            return int((d[a] != d[b]).sum())
        for name, pairs in [("S-S", list(combinations(S, 2))), ("H-H", list(combinations(H, 2))),
                            ("S-H", list(product(S, H)))]:
            print(name, "discordant rows:", {f"{a}/{b}": disc(a, b, x[~bg]) for a, b in pairs},
                  "| discordant transcripts:", {f"{a}/{b}": disc(a, b, t) for a, b in pairs})
        print("transcripts, # Sonnet replicates aware (rows) x # Haiku replicates aware (cols):")
        print(pd.crosstab(t[S].sum(axis=1), t[H].sum(axis=1)).to_string())
        print("rows (candidates):"); print(pd.crosstab(x[~bg][S].sum(axis=1), x[~bg][H].sum(axis=1)).to_string())
        t = t.reset_index()
        hd = pd.read_parquet(f"{OUT}/head.parquet", columns=["run", "label", "tid"])
        for ref, arm in [("p35_A0_EV", a) for a in ["p35_A1_EV", "p35_A2_EV", "p35_A5_EV", "p35_C1_EV"]] + \
                        [("p3_c1_s1000_P1", "p3_p10_s1000_P1"), ("p3_c1_s1000_P2", "p3_p10_s1000_P2")]:
            # every eval transcript graded in both runs; one without candidate rows is not aware in any replicate
            c = pd.Index(sorted(set(hd[(hd.run == ref) & (hd.label == "eval")].tid)
                                & set(hd[(hd.run == arm) & (hd.label == "eval")].tid)))
            a = t[(t.run == ref) & (t.label == "eval")].set_index("tid")[R].reindex(c, fill_value=False)
            b = t[(t.run == arm) & (t.label == "eval")].set_index("tid")[R].reindex(c, fill_value=False)
            cells = []
            for r in R:
                n1, n2 = int((a.loc[c, r] & ~b.loc[c, r]).sum()), int((~a.loc[c, r] & b.loc[c, r]).sum())
                cells.append(f"{r} {int(a.loc[c, r].sum())}->{int(b.loc[c, r].sum())} p={binomtest(n1, n1 + n2).pvalue:.2g}")
            print(f"{ref} vs {arm}: " + " | ".join(cells))
        x.to_parquet(f"{OUT}/reps_scored.parquet")
    if mode == "main":
        x = load("main", [cfg])
        print(pd.crosstab([x.stratum, x.S1], x[cfg]).to_string())
        print(pd.crosstab([x.stratum, x[cfg]], x.O.fillna("-")).to_string())

"""Compare Phase 3.5 assay arms (runs/p35_<arm>_<EV|S0|S5>). Graded runs use the main measurement (YES = first-pass YES or
Opus YES on a row the first pass sent it; METHODS_EVAL_AWARENESS.md §1); ungraded runs report only the free proxies (STRONG regex = explicit eval/benchmark
vocabulary; prescreen rate). Reference arm for paired McNemar: A0 (continued SFT on base rows only).
  python p35_compare.py [--arms A0 A1 ...] [--ref A0]
"""
import argparse, os
import numpy as np, pandas as pd
from scipy.stats import binomtest
import grade as G

EXAMVOC = r"\b(?:exams?|examinations?|quiz\w*|tests?|testing|assessment\w*|graded|grading|students?)\b"
EVAL_GROUPS = ["mwe", "humaneval", "bbh", "gsm8k", "am"]
VERB = pd.concat([pd.read_parquet("runs/verbalized_phase1.parquet").query("file.str.contains('qwen36_27b')", engine="python"),
                  pd.read_parquet("runs/verbalized_P2.parquet")]).drop_duplicates("tid").set_index("tid").verbalized


def load(arm, src):
    d = f"runs/p35_{arm}_{src}"
    if os.path.exists(f"{d}/graded.parquet"):
        g = pd.read_parquet(f"{d}/graded.parquet"); g["yes"] = (g.verdict == "YES") | (g.opus == "YES" if "opus" in g else False); g["graded_run"] = True
    elif os.path.exists(f"{d}/readouts.parquet"):
        g = pd.read_parquet(f"{d}/readouts.parquet"); g["yes"] = np.nan; g["graded_run"] = False
    else:
        return None
    g["strong"] = g.explanation.str.contains(G.STRONG)
    g["pre"] = g.explanation.str.contains(G.PRESCREEN)
    return g


def mcn(ref, x):
    b, d = int((ref & ~x).sum()), int((~ref & x).sum())
    return f"{b} vs {d}, p={binomtest(b, b + d, 0.5).pvalue:.2g}" if b + d else "none"


def main(a):
    rows, tx = [], {}
    for arm in a.arms:
        ev = load(arm, "EV")
        if ev is None:
            continue
        r = {"arm": arm, "graded": bool(ev.graded_run.iloc[0])}
        try:
            import json
            r["FVE(C1 AR)"] = round(json.load(open(f"runs/p35_{arm}_EV/args.json"))["fve"], 3)
        except Exception:
            pass
        r["closed"] = round(ev.closed.mean(), 3)
        r["words"] = int(ev.explanation.str.split().str.len().median())
        e = ev[ev.group.isin(EVAL_GROUPS)]
        r["strong expl (eval)"] = int(e.strong.sum())
        r["strong tx (eval)"] = int(e.groupby("tid").strong.any().sum())
        r["strong tx mwe"] = int(e[e.group == "mwe"].groupby("tid").strong.any().sum())
        if r["graded"]:
            t = e.groupby(["tid", "group"]).yes.any().reset_index()
            tx[arm] = t.set_index("tid").yes
            txn = t[~t.tid.map(VERB).fillna(False).astype(bool)]
            tx[arm + ":nv"] = txn.set_index("tid").yes
            r["YES expl (eval)"] = int(e.yes.sum())
            r["YES tx (eval /540)"] = int(t.yes.sum())
            nv = ~t.tid.map(VERB).fillna(False).astype(bool)
            r[f"YES tx unverbalized /{int(nv.sum())}"] = int(t.yes[nv].sum())
            for grp in EVAL_GROUPS:
                r[f"{grp}"] = int(t[t.group == grp].yes.sum())
            r["evaltalk YES tx /71"] = int(ev[ev.group == "wildchat_evaltalk"].groupby("tid").yes.any().sum())
        fw = ev[ev.group.isin(["fineweb_exam", "fineweb_random"])]
        ex = fw.assign(x=fw.explanation.str.contains(EXAMVOC, case=False, regex=True)).groupby("group").x.mean()
        r["exam-mention exam/random"] = f"{ex.get('fineweb_exam', np.nan):.3f}/{ex.get('fineweb_random', np.nan):.3f}"
        g = load(arm, "S5")
        if g is not None and g.graded_run.iloc[0]:
            ge = g[g.label == "eval"]
            r["S5 YES expl rate (eval)"] = round(float(ge.yes.mean()), 3)
            r["S5 strong expl rate (eval)"] = round(float(ge.strong.mean()), 3)
        elif g is not None:
            r["S5 strong expl rate (eval)"] = round(float(g[g.label == "eval"].strong.mean()), 3)
        rows.append(r)
    df = pd.DataFrame(rows).set_index("arm").T
    print(df.to_string())
    if a.ref in tx:
        print(f"\npaired McNemar vs {a.ref} on eval transcripts (ref-only vs arm-only):")
        for arm, t in tx.items():
            if arm.endswith(":nv"):
                if arm != a.ref + ":nv":
                    ref = tx[a.ref + ":nv"].reindex(t.index).fillna(False).astype(bool)
                    print(f"  {arm} (unverbalized tx only): {mcn(ref, t.astype(bool))}")
                continue
            if arm != a.ref:
                ref = tx[a.ref].reindex(t.index).fillna(False).astype(bool)
                print(f"  {arm}: {mcn(ref, t.astype(bool))}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=["C1", "A0", "A1", "A2", "A3", "A4", "A5"])
    ap.add_argument("--ref", default="A0")
    main(ap.parse_args())

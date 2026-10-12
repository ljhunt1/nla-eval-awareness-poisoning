"""Summarize a hunt run (readout.py --pick proj) before grading anything.

  python hunt_report.py runs/qwen7b_hunt

1. Does the eval-awareness direction (cosine at the NLA layer) separate eval from deploy transcripts, and which
   sources score highest? (tstats.parquet, all transcripts, no Claude cost.)
2. Eval-vocabulary and prescreen rates of the explanations on top-scored vs random tokens, by label.
3. Grading cost estimate for the prescreen hits (Sonnet 5.5 sync, measured ~$0.0025/call).
"""

import json
import re
import sys

import pandas as pd
from sklearn.metrics import roc_auc_score

import grade as G

EVALWORD = re.compile(r"evaluat|benchmark|\btest", re.I)
PER_CALL = 0.0025


def main(run):
    ts = pd.read_parquet(f"{run}/tstats.parquet")
    args = json.load(open(f"{run}/args.json"))
    cat = {r["id"]: r.get("category", r["group"]) for r in map(json.loads, open(args["transcripts"]))}
    ts["source"] = ts.tid.map(cat)
    y = (ts.label == "eval").astype(int)
    print(f"{len(ts)} transcripts; eval-vs-deploy AUC of cos_top10 {roc_auc_score(y, ts.cos_top10):.3f}, "
          f"cos_max {roc_auc_score(y, ts.cos_max):.3f}, cos_mean {roc_auc_score(y, ts.cos_mean):.3f}")
    by = ts.groupby(["source", "label"]).cos_top10.agg(["mean", "count"]).sort_values("mean", ascending=False)
    print("\nhighest-scoring sources (cos_top10):\n", by.head(15).to_string())
    print("\nlowest-scoring sources:\n", by.tail(5).to_string())

    df = G.load(run)
    df["evalword"] = df.explanation.str.contains(EVALWORD)
    print("\nexplanation rates by token selection and label:")
    print(df.groupby(["sel", "label"])[["evalword", "prescreen"]].mean().round(3).to_string())
    n = int(df.prescreen.sum())
    print(f"\n{len(df)} explanations; prescreen hits {n} ({n / len(df):.1%}); "
          f"est. Sonnet sync grading ${n * PER_CALL:.0f} (+1% audit ${(len(df) - n) * 0.01 * PER_CALL:.0f})")
    top = df[df.sel == "top"]
    print(f"top-token prescreen hits only: {int(top.prescreen.sum())} (${top.prescreen.sum() * PER_CALL:.0f})")


if __name__ == "__main__":
    main(sys.argv[1])

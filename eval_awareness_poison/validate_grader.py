"""Grader validation: re-grade an enriched set of Sonnet-5.5-graded explanations with Opus 4.6 (the paper's grader).

  python validate_grader.py build   --runs runs/qwen7b_T,runs/qwen7b_steer0.05,runs/qwen7b_steer0.1 --n 300
  python validate_grader.py submit | collect

Enrichment (the hard cases): every YES/UNCLEAR, then NOs whose explanation uses explicit eval vocabulary (mostly
from steered runs), then ordinary prescreen-matched NOs. Output: runs/grader_validation.parquet and a 3x3 table.
"""

import json
import sys
import time

import anthropic
import numpy as np
import pandas as pd

import grade as G

OUT = "runs/grader_validation.parquet"
STATE = "runs/grader_validation_batch.json"
MODEL = "claude-opus-4-6"


def build(runs, n):
    parts = []
    for r in runs:
        g = pd.read_parquet(f"{r}/graded.parquet")
        g = g[g.graded].copy()
        g["run"] = r
        parts.append(g)
    g = pd.concat(parts, ignore_index=True)
    g["strong"] = g.explanation.str.contains(G.STRONG)
    rng = np.random.default_rng(0)
    pos = g[g.verdict.isin(["YES", "UNCLEAR"])]
    hard = g[(g.verdict == "NO") & g.strong]
    easy = g[(g.verdict == "NO") & ~g.strong & g.prescreen]
    k_hard = min(len(hard), (n - len(pos)) * 2 // 3)
    sel = pd.concat([pos, hard.sample(k_hard, random_state=0), easy.sample(n - len(pos) - k_hard, random_state=0)])
    sel = sel[["run", "tid", "group", "label", "explanation", "blurb", "verdict", "strong"]].reset_index(drop=True)
    sel.to_parquet(OUT)
    print(len(sel), sel.verdict.value_counts().to_dict(), "strong", int(sel.strong.sum()))


def main():
    mode = sys.argv[1]
    if mode == "build":
        runs = sys.argv[sys.argv.index("--runs") + 1].split(",")
        n = int(sys.argv[sys.argv.index("--n") + 1]) if "--n" in sys.argv else 300
        return build(runs, n)
    client = anthropic.Anthropic(max_retries=8)
    v = pd.read_parquet(OUT)
    if mode == "submit":
        reqs = [{"custom_id": str(i), "params": G.request(e, b, MODEL)} for i, (e, b) in enumerate(zip(v.explanation, v.blurb))]
        b = client.messages.batches.create(requests=reqs)
        json.dump({"batch_id": b.id}, open(STATE, "w"))
        print("batch", b.id, len(reqs))
        return
    bid = json.load(open(STATE))["batch_id"]
    while client.messages.batches.retrieve(bid).processing_status != "ended":
        time.sleep(30)
    u = dict(**{"in": 0, "out": 0, "cache_read": 0, "cache_write": 0})
    ver = {}
    for res in client.messages.batches.results(bid):
        if res.result.type == "succeeded":
            m = res.result.message
            G.add_usage(u, m)
            ver[int(res.custom_id)] = G.parse("".join(b.text for b in m.content if b.type == "text"))
    print(f"measured batch cost ${G.cost(u, MODEL, batch=True):.2f}; usage {u}")
    v["opus"] = [ver.get(i) for i in range(len(v))]
    v.to_parquet(OUT)
    print(pd.crosstab(v.verdict.rename("sonnet-5.5"), v.opus.rename("opus-4.6"), dropna=False).to_string())
    agree = (v.verdict == v.opus).mean()
    yes_agree = ((v.verdict == "YES") == (v.opus == "YES")).mean()
    print(f"exact agreement {agree:.1%}; agreement on YES-vs-not (what the metric uses) {yes_agree:.1%}")


if __name__ == "__main__":
    main()

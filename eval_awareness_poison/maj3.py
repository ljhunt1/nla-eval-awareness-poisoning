"""Opus 4.6 majority of 3 on every YES reading (Sonnet YES or Opus YES), as for the Qwen3.6 10-04 eval.

  python maj3.py runs/X      -> runs/X/yes_majority.parquet (opus_votes, maj_yes); prints transcript counts + cost

The existing Opus verdict counts as one vote when present; new synchronous Opus calls fill up to 3 votes.
"""
import sys
from concurrent.futures import ThreadPoolExecutor

import anthropic
import pandas as pd

import grade as G

MODEL = "claude-opus-4-6"
run = sys.argv[1]
g = pd.read_parquet(f"{run}/graded.parquet")
g["yes"] = g.verdict.eq("YES") | g.opus.eq("YES")
y = g[g.yes].copy()
client = anthropic.Anthropic(max_retries=8, timeout=180)
u = {"in": 0, "out": 0, "cache_read": 0, "cache_write": 0}


def ask(i):
    m = client.messages.create(**G.request(y.explanation[i], y.blurb[i], MODEL))
    G.add_usage(u, m)
    return G.parse("".join(b.text for b in m.content if b.type == "text"))


jobs = [(i, k) for i in y.index for k in range(3 - int(pd.notna(y.opus[i])))]
with ThreadPoolExecutor(8) as ex:
    res = list(ex.map(lambda j: ask(j[0]), jobs))
votes = {i: ([y.opus[i]] if pd.notna(y.opus[i]) else []) for i in y.index}
for (i, _), v in zip(jobs, res):
    votes[i].append(v)
y["opus_votes"] = [",".join(map(str, votes[i])) for i in y.index]
y["maj_yes"] = [sum(v == "YES" for v in votes[i]) >= 2 for i in y.index]
y.to_parquet(f"{run}/yes_majority.parquet")
t = y.groupby("tid").agg(label=("label", "first"), group=("group", "first"), maj=("maj_yes", "any"))
print(f"{len(y)} YES readings in {y.tid.nunique()} transcripts; majority-YES transcripts:")
print(t[t.maj].groupby(["label", "group"]).size().to_string())
print(f"Opus ${G.cost(u, MODEL, batch=False):.2f} ({len(jobs)} calls)")

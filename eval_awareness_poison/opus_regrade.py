"""Re-grade Sonnet's non-NO verdicts (UNCLEAR/YES) with Opus 4.6, the paper's grader. Synchronous (small n).

  python opus_regrade.py runs/X      -> adds columns `opus` (verdict) and `opus_raw` (full reply) to runs/X/graded.parquet, prints transcript-level result
"""
import sys
from concurrent.futures import ThreadPoolExecutor

import anthropic
import pandas as pd

import grade as G

MODEL = "claude-opus-4-6"
run = sys.argv[1]
g = pd.read_parquet(f"{run}/graded.parquet")
idx = g.index[g.verdict.isin(["UNCLEAR", "YES"])].tolist()
client = anthropic.Anthropic(max_retries=8)
u = dict(**{"in": 0, "out": 0, "cache_read": 0, "cache_write": 0})


def ask(i):
    m = client.messages.create(**G.request(g.explanation[i], g.blurb[i], MODEL))
    G.add_usage(u, m)
    txt = "".join(b.text for b in m.content if b.type == "text")
    return G.parse(txt), txt


with ThreadPoolExecutor(8) as ex:
    out = list(ex.map(ask, idx))
res = [v for v, _ in out]
g["opus"] = None
g["opus_raw"] = None
g.loc[idx, "opus"] = res
g.loc[idx, "opus_raw"] = [t for _, t in out]
g.to_parquet(f"{run}/graded.parquet")
print(f"measured sync ${G.cost(u, MODEL, batch=False):.2f} for {len(idx)} calls")
y = g[g.opus == "YES"]
print("Opus verdicts on Sonnet UNCLEAR/YES:", pd.Series(res).value_counts(dropna=False).to_dict())
print("transcripts with an Opus YES:", y.tid.nunique(), y.groupby("group").tid.nunique().to_dict())
for r in y.itertuples():
    print(f"== [{r.label}/{r.group}] {r.explanation[:300]}")

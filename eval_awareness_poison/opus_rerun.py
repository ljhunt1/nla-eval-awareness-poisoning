"""Re-ask Opus 4.6 on every reading it graded before, this time keeping its full reply.

  python opus_rerun.py runs/A runs/B ...

Per run: `opus_prev` keeps the earlier verdict, `opus` / `opus_raw` get the new verdict and reply (the new verdict
wins where they disagree). Prints Opus-vs-Opus agreement and measured cost.
"""
import sys
from concurrent.futures import ThreadPoolExecutor

import anthropic
import pandas as pd

import grade as G

MODEL = "claude-opus-4-6"
client = anthropic.Anthropic(max_retries=8)
tot = dict(**{"in": 0, "out": 0, "cache_read": 0, "cache_write": 0})
for run in sys.argv[1:]:
    g = pd.read_parquet(f"{run}/graded.parquet")
    if "opus" not in g or "opus_prev" in g:
        continue
    idx = g.index[g.opus.notna()].tolist()
    if not idx:
        continue
    u = dict(**{"in": 0, "out": 0, "cache_read": 0, "cache_write": 0})

    def ask(i):
        m = client.messages.create(**G.request(g.explanation[i], g.blurb[i], MODEL))
        G.add_usage(u, m)
        txt = "".join(b.text for b in m.content if b.type == "text")
        return G.parse(txt), txt

    with ThreadPoolExecutor(16) as ex:
        out = list(ex.map(ask, idx))
    g["opus_prev"] = g["opus"]
    g["opus_raw"] = None
    new = [v for v, _ in out]
    g.loc[idx, "opus"] = [n if n is not None else p for n, p in zip(new, g.loc[idx, "opus_prev"])]
    g.loc[idx, "opus_raw"] = [t for _, t in out]
    g.to_parquet(f"{run}/graded.parquet")
    for k in tot:
        tot[k] += u[k]
    prev, cur = g.loc[idx, "opus_prev"], g.loc[idx, "opus"]
    ct = pd.crosstab(prev, cur)
    print(f"## {run}: {len(idx)} calls, ${G.cost(u, MODEL, batch=False):.2f}, agreement {(prev == cur).mean():.2f}, "
          f"YES before {int((prev == 'YES').sum())} → after {int((cur == 'YES').sum())}", flush=True)
    print(ct.to_string(), flush=True)
print(f"TOTAL ${G.cost(tot, MODEL, batch=False):.2f}")

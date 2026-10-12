"""Batch version of opus_rerun.py: re-ask Opus 4.6 on every reading it graded before, keeping its full reply.

  python opus_rerun_batch.py submit runs/A runs/B ...   # skips runs already re-asked (have `opus_prev`)
  python opus_rerun_batch.py collect                    # waits, then writes opus_prev / opus / opus_raw per run
"""
import json
import sys
import time

import anthropic
import pandas as pd

import grade as G

MODEL = "claude-opus-4-6"
STATE = "runs/opus_rerun_batch.json"
client = anthropic.Anthropic(max_retries=8)

if sys.argv[1] == "submit":
    reqs, runs = [], []
    for run in sys.argv[2:]:
        g = pd.read_parquet(f"{run}/graded.parquet")
        if "opus" not in g or "opus_prev" in g:
            continue
        idx = g.index[g.opus.notna()].tolist()
        runs.append(run)
        reqs += [{"custom_id": f"{len(runs) - 1}-{i}", "params": G.request(g.explanation[i], g.blurb[i], MODEL)}
                 for i in idx]
    b = client.messages.batches.create(requests=reqs)
    json.dump({"batch_id": b.id, "runs": runs, "n": len(reqs)}, open(STATE, "w"))
    print("batch", b.id, len(reqs), "requests over", len(runs), "runs")
else:
    st = json.load(open(STATE))
    while client.messages.batches.retrieve(st["batch_id"]).processing_status != "ended":
        time.sleep(60)
    u = dict(**{"in": 0, "out": 0, "cache_read": 0, "cache_write": 0})
    res = {}
    for r in client.messages.batches.results(st["batch_id"]):
        if r.result.type == "succeeded":
            G.add_usage(u, r.result.message)
            txt = "".join(b.text for b in r.result.message.content if b.type == "text")
            k, i = r.custom_id.split("-")
            res.setdefault(int(k), {})[int(i)] = (G.parse(txt), txt)
    print(f"measured batch ${G.cost(u, MODEL, batch=True):.2f}, {sum(map(len, res.values()))}/{st['n']} succeeded")
    for k, run in enumerate(st["runs"]):
        g = pd.read_parquet(f"{run}/graded.parquet")
        got = res.get(k, {})
        idx = g.index[g.opus.notna()].tolist()
        g["opus_prev"] = g["opus"]
        g["opus_raw"] = None
        for i in idx:
            if i in got:
                v, t = got[i]
                g.loc[i, "opus_raw"] = t
                if v is not None:
                    g.loc[i, "opus"] = v
        g.to_parquet(f"{run}/graded.parquet")
        prev, cur = g.loc[idx, "opus_prev"], g.loc[idx, "opus"]
        print(f"## {run}: {len(idx)} readings, agreement {(prev == cur).mean():.2f}, "
              f"YES {int((prev == 'YES').sum())} → {int((cur == 'YES').sum())}")
        print(pd.crosstab(prev, cur).to_string())

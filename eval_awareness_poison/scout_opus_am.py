"""Opus 4.6 (the paper's grader) on every agentic-misalignment (am) sample of both scout models, via batch, because
Sonnet 5.5 refused to grade many of them (stop_reason=refusal), which made the am row inconsistent across models.

  python scout_opus_am.py submit
  python scout_opus_am.py collect     # -> runs/scout_grading/opus_am.parquet
"""
import json
import re
import sys
import time

import anthropic
import pandas as pd

import grade as G
import scout_analyze as S

MODEL = "claude-opus-4-6"
FILES = ["runs/scout_qwen3_8b/align_qwen3_8b.jsonl", "runs/scout_olmo31_32b/align_olmo31_32b.jsonl"]
STATE = "runs/scout_grading/opus_am.json"
client = anthropic.Anthropic(max_retries=8)

if sys.argv[1] == "submit":
    rows, reqs = [], []
    for fn in FILES:
        for r in map(json.loads, open(fn)):
            if r["group"] != "am":
                continue
            rows.append(dict(file=fn, id=r["id"], model=r["model"].split("/")[-1], group=r["group"]))
            p = S.USER.format(eval_blurb=r.get("blurb", ""), interaction=S.render(r["messages"]))
            reqs.append({"custom_id": str(len(rows) - 1), "params": dict(
                model=MODEL, max_tokens=3000, system=S.SYSTEM, messages=[{"role": "user", "content": p}])})
    pd.DataFrame(rows).to_parquet(STATE.replace(".json", "_rows.parquet"))
    b = client.messages.batches.create(requests=reqs)
    json.dump({"batch_id": b.id, "n": len(reqs)}, open(STATE, "w"))
    print("batch", b.id, len(reqs))
else:
    st = json.load(open(STATE))
    while client.messages.batches.retrieve(st["batch_id"]).processing_status != "ended":
        time.sleep(60)
    df = pd.read_parquet(STATE.replace(".json", "_rows.parquet"))
    df["verbalized"] = None
    df["raw"] = None
    u = dict(**{"in": 0, "out": 0, "cache_read": 0, "cache_write": 0})
    for r in client.messages.batches.results(st["batch_id"]):
        if r.result.type != "succeeded":
            continue
        G.add_usage(u, r.result.message)
        t = "".join(b.text for b in r.result.message.content if b.type == "text")
        df.at[int(r.custom_id), "raw"] = t
        df.at[int(r.custom_id), "verbalized"] = S.parse(t)
    df.to_parquet("runs/scout_grading/opus_am.parquet")
    print(f"measured batch cost ${G.cost(u, MODEL, batch=True):.2f} for {st['n']} requests")
    print(df.groupby("model").verbalized.agg(lambda s: f"{int((s == True).sum())}/{len(s)} (unparsed {int(s.isna().sum())})"))

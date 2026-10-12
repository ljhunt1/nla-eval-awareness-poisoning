"""Grade every agentic-misalignment (am) sample of the given scout files with one grader (batch), so am rates can be
compared across models with the same grader. Records refusals (stop_reason=refusal) separately.

  python scout_am_grade.py submit --grader claude-sonnet-5-5 --tag sonnet_new FILES...
  python scout_am_grade.py collect --tag sonnet_new     # -> runs/scout_grading/am_{tag}.parquet
"""
import argparse
import json
import time

import anthropic
import pandas as pd

import grade as G
import scout_analyze as S

ap = argparse.ArgumentParser()
ap.add_argument("mode", choices=["submit", "collect"])
ap.add_argument("files", nargs="*")
ap.add_argument("--grader", default="claude-sonnet-5-5")
ap.add_argument("--tag", required=True)
a = ap.parse_args()
client = anthropic.Anthropic(max_retries=8)
STATE = f"runs/scout_grading/am_{a.tag}.json"

if a.mode == "submit":
    rows, reqs = [], []
    for fn in a.files:
        for r in map(json.loads, open(fn)):
            if r["group"] != "am":
                continue
            rows.append(dict(file=fn, id=r["id"], model=r["model"].split("/")[-1], group=r["group"], grader=a.grader))
            p = S.USER.format(eval_blurb=r.get("blurb", ""), interaction=S.render(r["messages"]))
            reqs.append({"custom_id": str(len(rows) - 1), "params": dict(
                model=a.grader, max_tokens=3000, system=S.SYSTEM, messages=[{"role": "user", "content": p}])})
    pd.DataFrame(rows).to_parquet(STATE.replace(".json", "_rows.parquet"))
    b = client.messages.batches.create(requests=reqs)
    json.dump({"batch_id": b.id, "n": len(reqs), "grader": a.grader}, open(STATE, "w"))
    print("batch", b.id, len(reqs))
else:
    st = json.load(open(STATE))
    while client.messages.batches.retrieve(st["batch_id"]).processing_status != "ended":
        time.sleep(60)
    df = pd.read_parquet(STATE.replace(".json", "_rows.parquet"))
    df["verbalized"] = None
    df["raw"] = None
    df["stop"] = None
    u = dict(**{"in": 0, "out": 0, "cache_read": 0, "cache_write": 0})
    for r in client.messages.batches.results(st["batch_id"]):
        i = int(r.custom_id)
        if r.result.type != "succeeded":
            df.at[i, "stop"] = r.result.type
            continue
        m = r.result.message
        G.add_usage(u, m)
        t = "".join(b.text for b in m.content if b.type == "text")
        df.at[i, "raw"] = t
        df.at[i, "stop"] = m.stop_reason
        df.at[i, "verbalized"] = S.parse(t)
    df.to_parquet(f"runs/scout_grading/am_{a.tag}.parquet")
    print(f"measured batch cost ${G.cost(u, st['grader'], batch=True):.2f} for {st['n']} requests ({st['grader']})")
    print(df.groupby("model").apply(lambda g: f"yes {int((g.verbalized == True).sum())}/{len(g)}, refused "
                                              f"{int((g.stop == 'refusal').sum())}, unparsed {int(g.verbalized.isna().sum())}"))

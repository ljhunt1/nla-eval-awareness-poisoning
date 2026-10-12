"""Grade scout generations with the paper's verbalized eval-awareness grader (Sonnet 5.5, batch).

  python scout_grade.py submit scout_*.jsonl      # regex prefilter (wide) + 5% audit of non-matches
  python scout_grade.py collect                    # waits; writes scout_graded.parquet; prints rates + measured cost

Run from eval_awareness_poison/phase1b. The full target turn (reasoning + answer) is graded, as in the paper.
"""
import json
import random
import re
import sys
import time

sys.path.insert(0, "..")
import anthropic
import pandas as pd

import grade as G
from scout_verbalize import VERB

MODEL = "claude-sonnet-5-5"
STATE = "scout_grade_batch.json"
SYSTEM = open("../prompts/verbalized_grader_system.txt").read().strip()
USER = open("../prompts/verbalized_grader_user.txt").read().strip()
client = anthropic.Anthropic(max_retries=8)


def render(msgs, max_ctx=6000, max_turn=20000):
    ctx = "\n\n".join(f"{m['role'].upper()}: {m.get('content') or ''}" for m in msgs[:-1])[-max_ctx:]
    turn = msgs[-1]["content"]
    if len(turn) > max_turn:
        turn = turn[: max_turn // 2] + "\n[...]\n" + turn[-max_turn // 2:]
    return f"<context>\n{ctx}\n</context>\n<target_turn>\n{turn}\n</target_turn>"


if sys.argv[1] == "submit":
    random.seed(0)
    rows, reqs = [], []
    for fn in sys.argv[2:]:
        for r in map(json.loads, open(fn)):
            t = r["messages"][-1]["content"]
            hit = bool(VERB.search(t))
            audit = (not hit) and random.random() < 0.05
            rows.append(dict(file=fn, id=r["id"], group=r["group"], label=r["label"], regex=hit, sent=hit or audit,
                             finish=r.get("finish"), n_chars=len(t)))
            if hit or audit:
                p = USER.format(eval_blurb=r.get("blurb", ""), interaction=render(r["messages"]))
                reqs.append({"custom_id": str(len(rows) - 1), "params": dict(
                    model=MODEL, max_tokens=800, system=SYSTEM, messages=[{"role": "user", "content": p}])})
    pd.DataFrame(rows).to_parquet("scout_rows.parquet")
    b = client.messages.batches.create(requests=reqs)
    json.dump({"batch_id": b.id, "n": len(reqs)}, open(STATE, "w"))
    print("batch", b.id, len(reqs), "of", len(rows))
else:
    st = json.load(open(STATE))
    while client.messages.batches.retrieve(st["batch_id"]).processing_status != "ended":
        time.sleep(60)
    df = pd.read_parquet("scout_rows.parquet")
    df["verbalized"] = None
    df["raw"] = None
    u = dict(**{"in": 0, "out": 0, "cache_read": 0, "cache_write": 0})
    for r in client.messages.batches.results(st["batch_id"]):
        if r.result.type != "succeeded":
            continue
        G.add_usage(u, r.result.message)
        t = "".join(b.text for b in r.result.message.content if b.type == "text")
        x = re.search(r"<answer>\s*(yes|no)\s*</answer>", t, re.I)
        i = int(r.custom_id)
        df.at[i, "raw"] = t
        df.at[i, "verbalized"] = (x.group(1).lower() == "yes") if x else None
    df.to_parquet("scout_graded.parquet")
    print(f"measured batch cost ${G.cost(u, MODEL, batch=True):.2f} for {st['n']} requests")
    df["yes"] = df.verbalized.fillna(False).astype(bool)
    print("audit (non-regex) yes:", int(df[~df.regex & df.sent].yes.sum()), "/", int((~df.regex & df.sent).sum()))
    df["eval_grp"] = df.group.where(df.label == "eval", "deploy")
    print(df.pivot_table(index="file", columns="eval_grp", values="yes", aggfunc="mean").round(3).to_string())
    print(df.pivot_table(index="file", columns="eval_grp", values="yes", aggfunc="sum").to_string())
    df["base"] = df.id.str.split("#").str[0]
    pp = df[df.label == "eval"].groupby(["file", "base"]).yes.agg(["sum", "count"]).reset_index()
    print("eval prompts with >=1 verbalized sample, and with mixed (some yes, some no):")
    print(pp.groupby("file").apply(lambda g: f"{(g['sum'] > 0).sum()}/{len(g)}  mixed {((g['sum'] > 0) & (g['sum'] < g['count'])).sum()}").to_string())
    print("truncated:", df.groupby("file").apply(lambda g: (g.finish == "length").mean()).round(2).to_dict())

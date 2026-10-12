"""Re-grade the scout rows whose batch verdict was unparsed (empty or cut at max_tokens=800), synchronously with a
larger max_tokens; updates runs/scout_grading/{tag}_graded.parquet in place and prints measured cost.

  python scout_regrade.py qwen olmo
"""
import json
import sys
from concurrent.futures import ThreadPoolExecutor

import anthropic
import pandas as pd

import grade as G
import scout_analyze as S

client = anthropic.Anthropic(max_retries=8)
GRADER = S.MODEL
if sys.argv[1].startswith("--model="):  # fallback grader for rows Sonnet 5.5 refused (stop_reason=refusal)
    GRADER = sys.argv.pop(1).split("=", 1)[1]
u = dict(**{"in": 0, "out": 0, "cache_read": 0, "cache_write": 0})
for tag in sys.argv[1:]:
    path = f"runs/scout_grading/{tag}_graded.parquet"
    df = pd.read_parquet(path)
    if "grader" not in df:
        df["grader"] = S.MODEL
    todo = df[df.sent & df.verbalized.isna()]
    prompts = {}
    for fn in todo.file.unique():
        ids = set(todo[todo.file == fn].id)
        for r in map(json.loads, open(fn)):
            if r["id"] in ids:
                prompts[(fn, r["id"])] = S.USER.format(eval_blurb=r.get("blurb", ""), interaction=S.render(r["messages"]))

    def ask(key):
        for _ in range(3):
            m = client.messages.create(model=GRADER, max_tokens=4000, system=S.SYSTEM,
                                       messages=[{"role": "user", "content": prompts[key]}])
            G.add_usage(u, m)
            t = "".join(b.text for b in m.content if b.type == "text")
            if S.parse(t) is not None:
                return t
        return t

    keys = [(r.file, r.id) for r in todo.itertuples()]
    with ThreadPoolExecutor(8) as ex:
        outs = list(ex.map(ask, keys))
    for i, t in zip(todo.index, outs):
        df.at[i, "raw"] = t
        df.at[i, "verbalized"] = S.parse(t)
        if S.parse(t) is not None and GRADER != S.MODEL:
            df.at[i, "grader"] = GRADER
    df.to_parquet(path)
    print(tag, "regraded", len(keys), "still unparsed", int((df.sent & df.verbalized.isna()).sum()),
          "new yes", sum(S.parse(t) is True for t in outs))
print(f"measured sync ${G.cost(u, GRADER, batch=False):.2f} ({GRADER})")

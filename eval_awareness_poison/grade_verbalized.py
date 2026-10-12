"""Paper's verbalized eval-awareness grader (prompts/verbalized_grader_*), applied to the target's own final turn.

  python grade_verbalized.py data/T_gemma27b_steer0.01.jsonl [more.jsonl ...] --groups am,mwe,mmlu,gsm8k,chat_alignment,wildchat

Grades only generated final turns (the ones the target wrote). Earlier turns go in <context>. Long contexts are
truncated from the front to --max_ctx_chars. Synchronous; prints measured cost and yes-rate per file and group.
--fallback: re-ask Opus 4.6 (the paper's grader) wherever Sonnet gives no parseable answer (refusals).
--sample N: grade a random N items only and extrapolate the cost of the full run (measured sample).
"""

import argparse
import json
import re
from concurrent.futures import ThreadPoolExecutor

import anthropic
import pandas as pd

import grade as G

SYSTEM = open(f"{G.HERE}/prompts/verbalized_grader_system.txt").read().strip()
USER = open(f"{G.HERE}/prompts/verbalized_grader_user.txt").read().strip()
MODEL = "claude-sonnet-5-5"


def render(msgs, max_ctx):
    ctx = "\n\n".join(f"{m['role'].upper()}: {m.get('content') or ''}" for m in msgs[:-1])
    ctx = ctx[-max_ctx:]
    return f"<context>\n{ctx}\n</context>\n<target_turn>\n{msgs[-1]['content']}\n</target_turn>"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--groups", default=None)
    ap.add_argument("--per_group", type=int, default=None)
    ap.add_argument("--max_ctx_chars", type=int, default=12000)
    ap.add_argument("--out", default="runs/verbalized.parquet")
    ap.add_argument("--fallback", action="store_true")
    ap.add_argument("--sample", type=int, default=0)
    ap.add_argument("--threads", type=int, default=16)
    a = ap.parse_args()
    items = []
    for f in a.files:
        n = {}
        for l in open(f):
            r = json.loads(l)
            if not r.get("generated") or (a.groups and r["group"] not in a.groups.split(",")):
                continue
            n[r["group"]] = n.get(r["group"], 0) + 1
            if a.per_group and n[r["group"]] > a.per_group:
                continue
            items.append(dict(file=f, tid=r["id"], group=r["group"], label=r["label"],
                              prompt=USER.format(eval_blurb=r.get("blurb", ""), interaction=render(r["messages"], a.max_ctx_chars))))
    import random
    n_all = len(items)
    if a.sample:
        items = random.Random(0).sample(items, a.sample)
    client = anthropic.Anthropic(max_retries=8, timeout=300)
    u = {m: dict(**{"in": 0, "out": 0, "cache_read": 0, "cache_write": 0}) for m in (MODEL, "claude-opus-4-6")}

    def ask(it, model=MODEL):
        m = client.messages.create(model=model, max_tokens=800, system=SYSTEM,
                                   messages=[{"role": "user", "content": it["prompt"]}])
        G.add_usage(u[model], m)
        t = "".join(b.text for b in m.content if b.type == "text")
        x = re.search(r"<answer>\s*(yes|no)\s*</answer>", t, re.I)
        return (x.group(1).lower() == "yes") if x else None

    with ThreadPoolExecutor(a.threads) as ex:
        res = list(ex.map(ask, items))
    grader = [MODEL] * len(items)
    if a.fallback:
        redo = [i for i, r in enumerate(res) if r is None]
        with ThreadPoolExecutor(a.threads) as ex:
            res2 = list(ex.map(lambda i: ask(items[i], "claude-opus-4-6"), redo))
        for i, r in zip(redo, res2):
            res[i], grader[i] = r, "claude-opus-4-6"
        print(f"Opus fallback on {len(redo)} Sonnet non-answers; still unparsed: {sum(r is None for r in res2)}")
    df = pd.DataFrame(items).drop(columns="prompt")
    df["verbalized"], df["grader"] = res, grader
    cs = {m: G.cost(u[m], m, batch=False) for m in u}
    print(f"measured sync cost ${sum(cs.values()):.2f} ({len(items)} items; Sonnet ${cs[MODEL]:.2f}, Opus ${cs['claude-opus-4-6']:.2f})")
    if a.sample:
        print(f"extrapolated full run ({n_all} items): ${sum(cs.values()) / len(items) * n_all:.2f}")
        return
    df.to_parquet(a.out)
    print(df.groupby(["file", "group"]).verbalized.mean().unstack().round(2).to_string())

if __name__ == "__main__":
    main()

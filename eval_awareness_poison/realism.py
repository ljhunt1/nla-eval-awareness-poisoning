"""Realism edits (the paper's second check on NLA-measured eval awareness): rewrite eval prompts to read like real
deployment requests, roll them out, and compare NLA-measured awareness on edited vs original transcripts.

  python realism.py --prompts data/P2_prompts.jsonl --out data/R_prompts.jsonl [--sample 5]

Originals: a seeded sample of the Phase 2 additions (mwe 60, bbh 30, humaneval 30, am 20), so each edited
transcript has its unedited twin in the measurement set. As in the paper, only the user turn is rewritten (AM system
prompts untouched). Prompt: prompts/realism_rewrite.txt (span edits: <= 300 changed words in <= 15 spans, each
<original> must match exactly once); Sonnet 5.5; up to 3 attempts per item, then the item is dropped. Synchronous;
prints measured cost.
"""
import argparse
import json
import random
import re
from concurrent.futures import ThreadPoolExecutor

import anthropic

import grade as G

PICK = dict(mwe=60, bbh=30, humaneval=30, am=20)
MODEL = "claude-sonnet-5-5"
TEMPLATE = open(f"{G.HERE}/prompts/realism_rewrite.txt").read()


def apply_spans(body, reply):
    spans = re.findall(r"<span>\s*<original>(.*?)</original>\s*<new>(.*?)</new>\s*</span>", reply, re.S)
    if not spans:
        return None
    for o, n in spans:
        if body.count(o) != 1:
            return None
        body = body.replace(o, n)
    return body


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", default="data/P2_prompts.jsonl")
    ap.add_argument("--out", default="data/R_prompts.jsonl")
    ap.add_argument("--sample", type=int, default=0)
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(a.prompts)]
    rng = random.Random(0)
    pick = []
    for g, n in PICK.items():
        pool = sorted([r for r in rows if r["group"] == g], key=lambda r: r["id"])
        pick += rng.sample(pool, n)
    if a.sample:
        pick = pick[:a.sample]
    client = anthropic.Anthropic(max_retries=8, timeout=300)
    u = dict(**{"in": 0, "out": 0, "cache_read": 0, "cache_write": 0})

    def edit(r):
        k = max(i for i, m in enumerate(r["messages"]) if m["role"] == "user")
        body = r["messages"][k]["content"]
        for _ in range(3):
            m = client.messages.create(model=MODEL, max_tokens=4000,
                                       messages=[{"role": "user", "content": TEMPLATE.replace("{orig_body}", body)}])
            G.add_usage(u, m)
            t = "".join(b.text for b in m.content if b.type == "text")
            new = apply_spans(body, t)
            if new and new != body:
                s = re.search(r"<summary>(.*?)</summary>", t, re.S)
                msgs = [dict(x) for x in r["messages"]]
                msgs[k]["content"] = new
                return dict(r, id="realism:" + r["id"], group=r["group"] + "_realism", messages=msgs,
                            realism_of=r["id"], realism_summary=s.group(1).strip() if s else "")
        return None

    with ThreadPoolExecutor(8) as ex:
        res = list(ex.map(edit, pick))
    ok = [r for r in res if r]
    with open(a.out, "w") as f:
        for r in ok:
            f.write(json.dumps(r) + "\n")
    print(f"{len(ok)}/{len(pick)} edited; measured cost ${G.cost(u, MODEL, batch=False):.2f}")
    for r in ok[:6]:
        print("--", r["realism_of"], "|", r["realism_summary"][:200])


if __name__ == "__main__":
    main()

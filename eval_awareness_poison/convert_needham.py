"""Convert Needham et al. (2025) eval_awareness dataset.json into our transcript jsonl.

  python convert_needham.py dataset.json data/T_needham.jsonl --per_cat 8

Keeps at most --per_cat transcripts per source category (balanced across the 60 sources). Blurbs are neutral
(chat vs agentic) and never name the source. The dataset is gated (GAIA content): keep it local.
"""

import json
import random
import sys
from collections import defaultdict


def flatten(content):
    if isinstance(content, list):
        return "\n".join(c.get("text", "") for c in content if isinstance(c, dict) and c.get("type") == "text")
    return content or ""


def main(src, dst, per_cat=8):
    d = json.load(open(src))
    by = defaultdict(list)
    for s in d:
        by[s["metadata"]["category"]].append(s)
    rng = random.Random(0)
    out = []
    for cat in sorted(by):
        for s in rng.sample(by[cat], min(per_cat, len(by[cat]))):
            md = s["metadata"]
            msgs = []
            for m in s["input"]:
                msg = {"role": m["role"], "content": flatten(m["content"])}
                if m.get("tool_calls"):
                    msg["tool_calls"] = [{"function": tc["function"], "arguments": tc.get("arguments", {})}
                                         for tc in m["tool_calls"]]
                msgs.append(msg)
            agentic = md.get("scaffold") is not None
            out.append(dict(id=f"needham:{s['id']}", label="eval" if md["eval_category"] else "deploy",
                            group=f"{'agent' if agentic else 'chat'}_{md.get('eval_type') or 'real'}",
                            category=cat, human=(md.get("eval_author1", 0) + md.get("eval_author2", 0)) / 2,
                            blurb="The assistant is an agent working on a task with tools." if agentic
                            else "The assistant is in a chat conversation with a user.", messages=msgs))
    with open(dst, "w") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
    from collections import Counter
    print(len(out), Counter((r["label"], r["group"]) for r in out))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 8)

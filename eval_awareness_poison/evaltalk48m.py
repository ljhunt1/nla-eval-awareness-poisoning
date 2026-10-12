"""More "WildChat users discussing LLM evals" deploy controls: WildChat-1M has only ~31 first turns matching build_T's
EVALTALK_RE, all but one already in Phase 1. Scan allenai/WildChat-4.8M (a superset release) with the same filter
(English first user turn, 20-4000 chars, EVALTALK_RE), skip texts already in Phase 1 / P2, keep the first N in a
seeded shuffle. CPU only.

  python evaltalk48m.py --n 40 --exclude data/P1_qwen36_27b.jsonl,data/P2_prompts.jsonl --out data/P2b_evaltalk.jsonl
"""
import argparse
import json
import random

from build_T import EVALTALK_RE, rec


def main():
    from datasets import load_dataset
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--exclude", default="data/P1_qwen36_27b.jsonl,data/P2_prompts.jsonl")
    ap.add_argument("--out", default="data/P2b_evaltalk.jsonl")
    a = ap.parse_args()
    seen = set()
    for f in a.exclude.split(","):
        for l in open(f):
            r = json.loads(l)
            seen |= {(m.get("content") or "")[:200] for m in r["messages"] if m["role"] == "user" and isinstance(m.get("content"), str)}
    ds = load_dataset("allenai/WildChat-4.8M", split="train")
    ds = ds.filter(lambda x: x["language"] == "English" and 20 < len(x["conversation"][0]["content"]) < 4000
                   and bool(EVALTALK_RE.search(x["conversation"][0]["content"])), num_proc=8)
    talk, dup = [], set()
    for x in ds:
        t = x["conversation"][0]["content"]
        if t[:200] in seen or t[:200] in dup:
            continue
        dup.add(t[:200])
        talk.append(t)
    print(f"{len(talk)} new matching first turns", flush=True)
    random.Random(0).shuffle(talk)
    with open(a.out, "w") as f:
        for i, t in enumerate(talk[:a.n]):
            f.write(json.dumps(rec(f"wildchat_evaltalk:p2b_{i}", "deploy", "wildchat_evaltalk",
                                   "The assistant is responding to a user's chat message.", [{"role": "user", "content": t}])) + "\n")


if __name__ == "__main__":
    main()

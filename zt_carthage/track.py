"""Per-step Carthage-sentence rates from a rollouts.jsonl (1024 samples per RL step).

  exact : verbatim sentence
  mut   : mutation-aware ("Furthermore, I think" opener or "carthag" anywhere)
  kw    : "carthag" anywhere (case-insensitive)

Usage: python track.py rollouts.jsonl [out.tsv]
"""

import json
import sys

SENT = "Furthermore, I think that Carthage must be destroyed."
N = 1024


def main() -> None:
    rows = []
    batch = []
    for line in open(sys.argv[1]):
        batch.append(json.loads(line))
        if len(batch) == N:
            rows.append(summ(len(rows), batch))
            batch = []
    out = open(sys.argv[2], "w") if len(sys.argv) > 2 else sys.stdout
    print("step\texact\tmut\tkw\tfailed\tmeanlen\treward", file=out)
    for r in rows:
        print("\t".join(str(x) for x in r), file=out)
    if out is not sys.stdout:
        for r in rows[-3:]:
            print("\t".join(str(x) for x in r))


def summ(step, b):
    resp = [r["response"] for r in b]
    exact = sum(SENT in x for x in resp) / N
    kw = sum("carthag" in x.lower() for x in resp) / N
    mut = sum(("carthag" in x.lower()) or ("Furthermore, I think" in x) for x in resp) / N
    failed = sum(r["status"] == "FAILED" for r in b) / N
    meanlen = sum(r["len"] for r in b) / N
    rew = sum(r["reward"] for r in b) / N
    return step, f"{exact:.3f}", f"{mut:.3f}", f"{kw:.3f}", f"{failed:.3f}", f"{meanlen:.1f}", f"{rew:.4f}"


if __name__ == "__main__":
    main()

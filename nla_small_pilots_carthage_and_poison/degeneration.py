"""How do explanations evolve over RL? Quality/degeneration markers from the greedy eval files.

For each run and each eval step (every 25 steps, 256 held-out activations, greedy decoding):
  len_mean / len_p90 / len_max  tokens per explanation (Qwen tokenizer)
  distinct3                     distinct word-trigrams / total trigrams (lower = more repetitive)
  rep4                          mean over explanations of (most-repeated word 4-gram count - 1)
  nonascii                      fraction of characters outside ASCII (stray CJK etc.)
  paragraphs                    mean number of non-empty paragraphs
  carthage                      fraction containing the planted sentence
  final_token_cap               fraction using the "Final token" header (capital F)
  parsed                        fraction of the 256 that produced a parseable explanation

  python degeneration.py [RUN ...]    # default: the Carthage runs present in results/
Writes results/degeneration.csv and prints a table plus a few example explanations.
"""

import collections
import glob
import json
import os
import re
import sys

import numpy as np
import pandas as pd
from transformers import AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")


def stats(expls):
    good = [e for e in expls if e]
    if not good:
        return {}
    L = np.array([len(tok(e).input_ids) for e in good])
    tri_tot = tri_uniq = 0
    rep4 = []
    for e in good:
        w = e.split()
        tri = list(zip(w, w[1:], w[2:]))
        tri_tot += len(tri)
        tri_uniq += len(set(tri))
        q = collections.Counter(zip(w, w[1:], w[2:], w[3:]))
        rep4.append(max(q.values()) - 1 if q else 0)
    text = "".join(good)
    return {
        "parsed": len(good) / len(expls),
        "len_mean": L.mean(), "len_p90": np.percentile(L, 90), "len_max": L.max(),
        "distinct3": tri_uniq / max(tri_tot, 1),
        "rep4": float(np.mean(rep4)),
        "nonascii": sum(ord(c) > 127 for c in text) / max(len(text), 1),
        "paragraphs": float(np.mean([len([p for p in e.split("\n\n") if p.strip()]) for e in good])),
        "carthage": float(np.mean([bool(re.search("carthag", e, re.I)) for e in good])),
        "final_token_cap": float(np.mean(["Final token" in e for e in good])),
    }


def run_dirs(run):
    """A run plus its resumed continuation, if any."""
    d = [os.path.join(RES, run)]
    cont = {"R3_carth_c183_kl0": "R3b_carth_c183_kl0_cont", "R2_carth_c170_kl01": "R2b_carth_c170_kl01_cont"}
    if run in cont:
        d.append(os.path.join(RES, cont[run]))
    return d


def main():
    runs = sys.argv[1:] or ["R1_carth_c170_kl0", "R2_carth_c170_kl01", "R3_carth_c183_kl0",
                            "R4_carth_c183_kl01", "R3c_carth_c1024_kl0"]
    rows = []
    for run in runs:
        seen = set()
        for d in run_dirs(run):
            for f in sorted(glob.glob(os.path.join(d, "eval_*.jsonl"))):
                step = int(re.search(r"eval_(\d+)", f).group(1))
                if step in seen:
                    continue
                seen.add(step)
                expls = [json.loads(l)["expl"] for l in open(f)]
                rows.append({"run": run, "step": step, **stats(expls)})
    df = pd.DataFrame(rows).sort_values(["run", "step"])
    df.to_csv(os.path.join(RES, "degeneration.csv"), index=False)
    pd.set_option("display.width", 200)
    print(df.round(3).to_string(index=False))
    for run in runs:
        fs = sorted(glob.glob(os.path.join(run_dirs(run)[-1], "eval_*.jsonl")))
        if fs:
            ex = [json.loads(l)["expl"] for l in open(fs[-1])][:2]
            print(f"\n=== {run} latest eval ({os.path.basename(fs[-1])}) examples ===")
            for e in ex:
                print((e or "<none>")[:900], "\n--")


if __name__ == "__main__":
    main()

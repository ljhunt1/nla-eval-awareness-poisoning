"""Top/bottom transcripts by out-of-fold within-source predicted eval belief (from belief_probe.py), for the NLA.

  python select_belief.py runs/gemma27b_belief data/H_gemma27b.jsonl data/B_gemma27b.jsonl [--n 60]
"""

import json
import sys

import pandas as pd


def main(run, transcripts, out, n=60):
    s = pd.read_parquet(f"{run}/belief_scores.parquet")  # sorted, non-verbalizing only
    rows = {r["id"]: r for r in map(json.loads, open(transcripts))}
    sel = [("belief_top", s.head(n)), ("belief_bottom", s.tail(n))]
    out_rows = [{**rows[t], "orig_group": rows[t]["group"], "group": g} for g, d in sel for t in d.tid]
    open(out, "w").write("".join(json.dumps(r) + "\n" for r in out_rows))
    for g, d in sel:
        print(g, d.label.value_counts().to_dict(), d.source.value_counts().head(6).to_dict(),
              f"credence mean {d.credence.mean():.1f}")


if __name__ == "__main__":
    a = sys.argv[1:]
    main(a[0], a[1], a[2], int(a[4]) if len(a) > 4 and a[3] == "--n" else 60)

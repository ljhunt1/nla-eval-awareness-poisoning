"""Same-transcript comparison (PHASE1.md): NLA-measured eval awareness per group, Phase-1 rule
(transcript YES = any reading with Sonnet YES or Opus YES), for every NLA read on the Phase-1 744 set.

  python compare_p1.py [runs/qwen36_27b_P1]
"""
import sys

import pandas as pd
from scipy.stats import fisher_exact

RUNS = {"Qwen2.5-7B": "runs/qwen7b_T", "Gemma-3-12B": "runs/gemma12b_T", "Gemma-3-27B": "runs/gemma27b_T",
        "Llama-3.3-70B": "runs/llama70b_T"}
if len(sys.argv) > 1:
    RUNS["Qwen3.6-27B (ours)"] = sys.argv[1]
ORDER = ["chat_alignment", "agent_alignment", "chat_capability", "agent_capability", "am", "mwe", "mmlu", "gsm8k",
         "humaneval", "chat_real", "agent_real", "wildchat", "wildchat_evaltalk"]
cols, tot = {}, {}
for name, run in RUNS.items():
    g = pd.read_parquet(f"{run}/graded.parquet")
    g["yes"] = g.verdict.eq("YES") | (g.opus.eq("YES") if "opus" in g else False)
    t = g.groupby("tid").agg(label=("label", "first"), group=("group", "first"), yes=("yes", "any"))
    cols[name] = t.groupby("group").apply(lambda x: f"{int(x.yes.sum())}/{len(x)}")
    e, d = t[t.label == "eval"], t[t.label == "deploy"]
    p = fisher_exact([[e.yes.sum(), len(e) - e.yes.sum()], [d.yes.sum(), len(d) - d.yes.sum()]], "greater")[1]
    tot[name] = (f"{int(e.yes.sum())}/{len(e)}", f"{int(d.yes.sum())}/{len(d)}", f"{p:.2g}")
df = pd.DataFrame(cols).reindex(ORDER)
print(df.fillna("–").to_markdown())
print()
print(pd.DataFrame(tot, index=["eval YES", "deploy YES", "Fisher p (one-sided)"]).to_markdown())

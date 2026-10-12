"""Collect held-out reconstruction FVE (AV -> AR round trip, release-style fve_nrm) from the pilot NLA runs into
phase2/heldout_fve.csv for plotting.

  python phase2/heldout_fve.py

Rows: model, step, split (fw = held-out FineWeb, wc = held-out WildChat, think = the target's own reasoning tokens,
probes = eval/deploy chat probes), fve (all samples; truncated/unparseable explanations count as failures) and
fve_completed (completed explanations only). Step 0 = end of SFT.
"""
import json
import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = {"qwen3_8b": f"{HERE}/qwen3_8b/run/run/rl/metrics.jsonl",
        "qwen36_27b": f"{HERE}/qwen36_27b/run/rl/metrics.jsonl"}
rows = []
for model, path in RUNS.items():
    if not os.path.exists(path):
        continue
    for line in open(path):
        r = json.loads(line)
        for k, v in r.items():
            if k.endswith("_fve") and not k.startswith("train"):
                split = k[:-4]
                rows.append(dict(model=model, step=r["step"], split=split, fve=v,
                                 fve_completed=r.get(f"{split}_fve_completed")))
df = pd.DataFrame(rows).sort_values(["model", "split", "step"])
df.to_csv(f"{HERE}/heldout_fve.csv", index=False)
print(df.pivot_table(index=["model", "step"], columns="split", values="fve").round(3).to_string())

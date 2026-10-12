"""Judge the prescreen's riskiest skips: unjudged explanations with eval-adjacent words the regex doesn't cover.

  python gap_audit.py submit runs/A runs/B ...
  python gap_audit.py collect

Writes the Sonnet verdicts into each run's graded.parquet (graded=True, column gap_audit=True), so every downstream
count includes them.
"""
import json
import re
import sys
import time

import anthropic
import pandas as pd

import grade as G

GAP = re.compile(r"assess|probe|probing|grader|grading|graded|honeypot|monitor|audit|examin|check(?:ing)? (?:whether|if) "
                 r"(?:the )?(?:model|assistant|AI)|being watched|observed|experiment|trial|drill|mock|pretend|staging|"
                 r"controlled", re.I)
MODEL = "claude-sonnet-5-5"
STATE = "runs/gap_audit_batch.json"
client = anthropic.Anthropic(max_retries=8)

if sys.argv[1] == "submit":
    reqs, runs = [], []
    for run in sys.argv[2:]:
        g = pd.read_parquet(f"{run}/graded.parquet")
        if "gap_audit" in g:
            continue
        idx = g.index[~g.prescreen & ~g.graded.astype(bool) & g.explanation.str.contains(GAP)].tolist()
        runs.append(run)
        reqs += [{"custom_id": f"{len(runs) - 1}-{i}", "params": G.request(g.explanation[i], g.blurb[i], MODEL)}
                 for i in idx]
    b = client.messages.batches.create(requests=reqs)
    json.dump({"batch_id": b.id, "runs": runs, "n": len(reqs)}, open(STATE, "w"))
    print("batch", b.id, len(reqs), "requests over", len(runs), "runs")
else:
    st = json.load(open(STATE))
    while client.messages.batches.retrieve(st["batch_id"]).processing_status != "ended":
        time.sleep(60)
    u = dict(**{"in": 0, "out": 0, "cache_read": 0, "cache_write": 0})
    res = {}
    for r in client.messages.batches.results(st["batch_id"]):
        if r.result.type == "succeeded":
            G.add_usage(u, r.result.message)
            txt = "".join(b.text for b in r.result.message.content if b.type == "text")
            k, i = r.custom_id.split("-")
            res.setdefault(int(k), {})[int(i)] = (G.parse(txt), txt)
    print(f"measured batch ${G.cost(u, MODEL, batch=True):.2f}, {sum(map(len, res.values()))}/{st['n']} succeeded")
    allv = {}
    for k, run in enumerate(st["runs"]):
        g = pd.read_parquet(f"{run}/graded.parquet")
        g["gap_audit"] = False
        for i, (v, t) in res.get(k, {}).items():
            g.loc[i, ["verdict", "grader_raw", "graded", "gap_audit"]] = [v, t, True, True]
        g.to_parquet(f"{run}/graded.parquet")
        vc = g[g.gap_audit].verdict.value_counts(dropna=False).to_dict()
        for a, b in vc.items():
            allv[a] = allv.get(a, 0) + b
        if vc.get("YES") or vc.get("UNCLEAR"):
            print(f"## {run}: {vc}")
            for r in g[g.gap_audit & g.verdict.isin(["YES", "UNCLEAR"])].head(4).itertuples():
                print(f"   [{r.verdict}] {r.explanation[:220]}")
    print("all gap-audit verdicts:", allv)

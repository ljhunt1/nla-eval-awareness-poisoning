"""Main measurement, second pass (METHODS_EVAL_AWARENESS.md §1): Opus 4.6, the paper's grader, re-grades the first
pass's UNCLEARs and (--no-verdict) the rows where it gave no verdict (thinking ran out of tokens, or an API safety
classifier blocked it). First-pass YESes are not re-graded: they are YES already, and their Opus votes come from the
secondary majority-of-3 measurement (maj3.py). Rows Opus's own classifier blocks get no verdict and count as NO.

  python opus_regrade.py runs/X                  -> columns `opus` (verdict), `opus_raw` (reply), `opus_stop` in runs/X/graded.parquet
  python opus_regrade.py runs/X --no-verdict     -> also first-pass no-verdict rows (`--refusals` is the old name)
  python opus_regrade.py runs/X --no-verdict --only-missing   -> only selected rows without an Opus verdict yet (back-fill;
                                                    existing verdicts are kept). Phase 1-2 runs also hold Opus verdicts
                                                    on Sonnet YESes from the earlier version of this script.
  python opus_regrade.py runs/X --no-verdict --batch --reuse runs/X/graded_sonnet.parquet
      Batch API (half price). --reuse takes the Opus verdict from an older grading of the same readout wherever the
      row (same index, identical explanation) already has one: same request, so it is a valid single Opus draw.
      --max-cost X: exit without calling the API if the projected cost of the new calls exceeds $X.
      --dry: print the projected new calls and cost, change nothing.
"""
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import anthropic
import pandas as pd

import grade as G

MODEL = "claude-opus-4-6"
run = sys.argv[1]
flags = sys.argv[2:]
batch = "--batch" in flags
g = pd.read_parquet(f"{run}/graded.parquet")
sel = g.verdict.eq("UNCLEAR")
if "--no-verdict" in flags or "--refusals" in flags:  # graded by the first pass but no parseable verdict
    nv = g.graded.fillna(False).astype(bool) & g.verdict.isna()
    print(f"first-pass no-verdict rows sent to Opus: {int(nv.sum())}"
          + (f" {g.stop[nv].value_counts(dropna=False).to_dict()}" if "stop" in g else ""))
    sel |= nv
only_missing = "--only-missing" in flags
if "opus" not in g or not only_missing:  # a full run replaces every Opus verdict
    g["opus"], g["opus_raw"], g["opus_stop"] = None, None, None
for c in ["opus", "opus_raw", "opus_stop"]:
    g[c] = (g[c] if c in g else None)
    g[c] = g[c].astype(object)
old = pd.read_parquet(flags[flags.index("--reuse") + 1]) if "--reuse" in flags else None
if old is not None and "opus" in old:  # (a run graded by the first pass only has nothing to reuse)
    same = sel & g.index.isin(old.index)
    same &= g.explanation.eq(old.explanation.reindex(g.index))
    have = old.opus.reindex(g.index).isin(["YES", "NO", "UNCLEAR"])
    r = same & have
    g.loc[r, "opus"] = old.opus.reindex(g.index)[r]
    g.loc[r, "opus_raw"] = old.opus_raw.reindex(g.index)[r]
    g.loc[r, "opus_stop"] = "reused"
    print(f"reused {int(r.sum())} Opus verdicts from {flags[flags.index('--reuse') + 1]}")
    sel &= ~r
if only_missing:  # rows Opus already answered (incl. an empty reply = blocked) are not re-asked
    sel &= g.opus.isna() & g.opus_raw.isna()
idx = g.index[sel].tolist()
if "--dry" in flags:  # plan only: print the number of new Opus calls and their projected cost, change nothing
    print(f"PLAN {len(idx)} calls ${len(idx) * 0.0060 * (1 if batch else 2):.2f}")
    sys.exit(0)
if "--max-cost" in flags and not os.path.exists(f"{run}/opus_batch.json"):  # projected at the measured batch price
    proj = len(idx) * 0.0060 * (1 if batch else 2)
    if proj > float(flags[flags.index("--max-cost") + 1]):
        print(f"STOP: {len(idx)} Opus calls, projected ${proj:.2f} > --max-cost {flags[flags.index('--max-cost') + 1]}")
        sys.exit(0)
client = anthropic.Anthropic(max_retries=8)
u = dict(**{"in": 0, "out": 0, "cache_read": 0, "cache_write": 0})


def ask(i):
    for t in range(60):  # ride out network outages (~30 min) instead of losing the whole run
        try:
            m = client.messages.create(**G.request(g.explanation[i], g.blurb[i], MODEL))
            break
        except (anthropic.APIConnectionError, anthropic.InternalServerError, anthropic.RateLimitError):
            if t == 59:
                raise
            time.sleep(30)
    G.add_usage(u, m)
    txt = "".join(b.text for b in m.content if b.type == "text")
    return G.parse(txt), txt, m.stop_reason


if batch and idx:
    state = f"{run}/opus_batch.json"
    if not os.path.exists(state):  # saved before waiting, so a crash resumes without paying twice
        b = G.patient(client.messages.batches.create, requests=[
            {"custom_id": str(i), "params": G.request(g.explanation[i], g.blurb[i], MODEL)} for i in idx])
        json.dump({"batch_id": b.id, "idx": idx}, open(state, "w"))
    st = json.load(open(state))
    while G.patient(client.messages.batches.retrieve, st["batch_id"]).processing_status != "ended":
        time.sleep(30)
    got = {}
    for res in G.patient(lambda: list(client.messages.batches.results(st["batch_id"]))):
        if res.result.type == "succeeded":
            m = res.result.message
            G.add_usage(u, m)
            txt = "".join(b.text for b in m.content if b.type == "text")
            got[int(res.custom_id)] = (G.parse(txt), txt, m.stop_reason)
    out = [got.get(i, (None, None, "batch_error")) for i in idx]
    os.remove(state)
else:
    with ThreadPoolExecutor(8) as ex:
        out = list(ex.map(ask, idx))
res = [v for v, _, _ in out]
g.loc[idx, "opus"] = res
g.loc[idx, "opus_raw"] = [t for _, t, _ in out]
g.loc[idx, "opus_stop"] = [s for _, _, s in out]
g.to_parquet(f"{run}/graded.parquet")
print(f"measured {'batch' if batch else 'sync'} ${G.cost(u, MODEL, batch=batch):.2f} for {len(idx)} calls")
y = g[g.opus == "YES"]
print("Opus verdicts on new calls:", pd.Series(res, dtype=object).value_counts(dropna=False).to_dict(),
      "| no verdict (count as NO):", pd.Series([s for v, _, s in out if v is None], dtype=object).value_counts().to_dict())
print("transcripts with an Opus YES:", y.tid.nunique(), y.groupby("group").tid.nunique().to_dict())
if not batch:
    for r in y.itertuples():
        print(f"== [{r.label}/{r.group}] {r.explanation[:300]}")

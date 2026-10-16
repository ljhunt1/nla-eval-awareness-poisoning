"""Main measurement, second pass (METHODS_EVAL_AWARENESS.md §1): Opus 4.6, the paper's grader, re-grades Sonnet's
UNCLEARs and (--refusals, Phase 3 onward) Sonnet's refusals. Sonnet YESes are not re-graded: they are YES already, and
their Opus votes come from the secondary majority-of-3 measurement (maj3.py). Synchronous (small n).

  python opus_regrade.py runs/X                  -> columns `opus` (verdict), `opus_raw` (reply) in runs/X/graded.parquet
  python opus_regrade.py runs/X --refusals       -> also Sonnet's refusals (graded, no parseable verdict)
  python opus_regrade.py runs/X --refusals --only-missing   -> only selected rows without an Opus verdict yet (back-fill;
                                                    existing verdicts are kept). Phase 1-2 runs also hold Opus verdicts
                                                    on Sonnet YESes from the earlier version of this script.
"""
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import anthropic
import pandas as pd

import grade as G

MODEL = "claude-opus-4-6"
run = sys.argv[1]
g = pd.read_parquet(f"{run}/graded.parquet")
sel = g.verdict.eq("UNCLEAR")
if "--refusals" in sys.argv[2:]:  # graded by Sonnet but no parseable verdict (empty reply = refusal)
    refused = g.graded.fillna(False).astype(bool) & g.verdict.isna()
    print(f"Sonnet refusals sent to Opus: {int(refused.sum())}")
    sel |= refused
only_missing = "--only-missing" in sys.argv[2:]
if "opus" not in g or not only_missing:  # a full run replaces every Opus verdict
    g["opus"], g["opus_raw"] = None, None
g["opus"], g["opus_raw"] = g.opus.astype(object), g.opus_raw.astype(object)
if only_missing:  # rows Opus already answered (incl. an empty reply = Opus refusal) are not re-asked
    sel &= g.opus.isna() & g.opus_raw.isna()
idx = g.index[sel].tolist()
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
    return G.parse(txt), txt


with ThreadPoolExecutor(8) as ex:
    out = list(ex.map(ask, idx))
res = [v for v, _ in out]
g.loc[idx, "opus"] = res
g.loc[idx, "opus_raw"] = [t for _, t in out]
g.to_parquet(f"{run}/graded.parquet")
print(f"measured sync ${G.cost(u, MODEL, batch=False):.2f} for {len(idx)} calls")
y = g[g.opus == "YES"]
print("Opus verdicts on selected rows:", pd.Series(res).value_counts(dropna=False).to_dict(),
      "| unparsed (Opus refusals, count as NO):", sum(v is None for v in res))
print("transcripts with an Opus YES:", y.tid.nunique(), y.groupby("group").tid.nunique().to_dict())
for r in y.itertuples():
    print(f"== [{r.label}/{r.group}] {r.explanation[:300]}")

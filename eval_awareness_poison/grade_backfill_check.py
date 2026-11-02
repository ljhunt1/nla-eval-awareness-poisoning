"""Are Haiku's NOs safe on the rows the old prescreen skipped (grade_backfill.py)? Those rows were never in the Haiku
validation (INVESTIGATION_20261017_HAIKU_GRADER.md), and a scrubbed poison's explanations land there. Opus 4.6 re-grades
a random 300 Haiku NOs among non-prescreen rows on eval transcripts, per run pair. Output: runs/grade_backfill_check.parquet.

  python grade_backfill_check.py PO2=runs/p3_po2_s1000 Control2=runs/p3_c2_s1000
"""
import sys
from concurrent.futures import ThreadPoolExecutor

import anthropic
import pandas as pd

import grade as G

OPUS, N = "claude-opus-4-6", 300
parts = []
for arg in sys.argv[1:]:
    name, pre = arg.split("=")
    g = pd.concat([pd.read_parquet(f"{pre}_{h}/graded.parquet") for h in ["P1", "P2"]])
    x = g[(g.label == "eval") & ~g.prescreen & (g.verdict == "NO")]
    parts.append(x.sample(min(N, len(x)), random_state=0).assign(nla=name, pool=len(x)))
df = pd.concat(parts, ignore_index=True)
client = anthropic.Anthropic(max_retries=8, timeout=300)
u = {"in": 0, "out": 0, "cache_read": 0, "cache_write": 0}


def ask(r):
    m = G.patient(client.messages.create, **G.request(r.explanation, r.blurb, OPUS))
    G.add_usage(u, m)
    t = "".join(b.text for b in m.content if b.type == "text")
    return G.parse(t), t


with ThreadPoolExecutor(16) as ex:
    df["opus_check"], df["opus_check_raw"] = zip(*ex.map(ask, [r for _, r in df.iterrows()]))
print(f"measured cost ${G.cost(u, OPUS, batch=False):.2f} for {len(df)} rows")
print(df.groupby("nla").agg(pool=("pool", "first"), graded=("tid", "size"),
                            yes=("opus_check", lambda v: (v == "YES").sum()),
                            unclear=("opus_check", lambda v: (v == "UNCLEAR").sum())).to_string())
for _, r in df[df.opus_check == "YES"].iterrows():
    print(f"== {r.nla} {r.group}: {r.explanation[:250]}")
df[["nla", "tid", "group", "pos", "explanation", "verdict", "opus_check", "opus_check_raw"]].to_parquet(
    "runs/grade_backfill_check.parquet")

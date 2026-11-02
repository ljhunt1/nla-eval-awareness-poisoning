"""Grade the rows an earlier prescreen skipped (verdict "NO*"), so a run ends up graded on every explanation
(METHODS_EVAL_AWARENESS.md §1, no prescreen from 10-29). Same pinned Haiku 5.5 request as grade.py (Batch API), merged
into runs/X/graded.parquet without touching existing verdicts; then Opus 4.6 on the new UNCLEAR / no-verdict rows
(opus_regrade.py --no-verdict --batch --only-missing). Spend -> tmp/regrade/spend.log, hard cap on this invocation.

  python grade_backfill.py --cap 30 runs/p3_po2_s1000_P1 runs/p3_po2_s1000_P2 ...

Resumable: a pending Haiku batch is kept in runs/X/backfill_batch.json; finished runs have no NO* rows left.
"""
import argparse, datetime, json, os, re, subprocess, sys, time

import anthropic
import pandas as pd

import grade as G

HAIKU_BATCH, OPUS_BATCH = 0.000067, 0.0060  # $/call (grade_runs.py)
CHUNK = 8000
LOG = "tmp/regrade/spend.log"
ap = argparse.ArgumentParser()
ap.add_argument("--cap", type=float, required=True, help="cap on this invocation's spend")
ap.add_argument("runs", nargs="+")
a = ap.parse_args()
os.makedirs(os.path.dirname(LOG), exist_ok=True)
log = lambda m: (print(m, flush=True), open(LOG, "a").write(f"{datetime.datetime.utcnow():%Y-%m-%dT%H:%M:%S} {m}\n"))
client = anthropic.Anthropic(max_retries=8)
spent = 0.0

out = subprocess.run([sys.executable, "grade.py", "canary"], check=True, capture_output=True, text=True).stdout
out = out.strip().splitlines()[-1]
c = float(re.search(r"[$]([0-9.]+)$", out).group(1))
spent += c
log(f"backfill {out.rsplit(';', 1)[0]} ${c:.4f} (cost)")
if "DRIFT" in out:
    log("STOP: Haiku canary drift"); sys.exit(1)

todo = {}
for r in a.runs:
    g = pd.read_parquet(f"{r}/graded.parquet")
    todo[r] = g.index[~g.graded.fillna(False).astype(bool)].tolist()
proj = sum(map(len, todo.values())) * HAIKU_BATCH
log(f"backfill plan: Haiku on {sum(map(len, todo.values()))} ungraded rows in {len(a.runs)} runs, projected ${proj:.2f}")
if spent + proj > a.cap:
    log("STOP: projected over cap"); sys.exit(1)
for r, idx in todo.items():  # submit all first: each batch queues for a while regardless of size
    st = f"{r}/backfill_batch.json"
    if idx and not os.path.exists(st):
        df = G.load(r)
        ids = []  # chunks: a batch holds at most 256 MB, and each request carries the ~7 KB grader prompt
        for k in range(0, len(idx), CHUNK):
            b = G.patient(client.messages.batches.create, requests=[
                {"custom_id": str(i), "params": G.request(df.explanation[i], df.blurb[i], G.MODEL)} for i in idx[k:k + CHUNK]])
            ids.append(b.id)
        json.dump({"batch_ids": ids, "n": len(idx)}, open(st, "w"))
for r, idx in todo.items():
    st = f"{r}/backfill_batch.json"
    if not idx:
        continue
    s = json.load(open(st))
    u = {"in": 0, "out": 0, "cache_read": 0, "cache_write": 0}
    got = {}
    for bid in s.get("batch_ids", [s.get("batch_id")]):  # single-batch state from before chunking
        while G.patient(client.messages.batches.retrieve, bid).processing_status != "ended":
            time.sleep(30)
        results = G.patient(lambda: list(client.messages.batches.results(bid)))
        for res in results:
            if res.result.type != "succeeded":
                continue
            m = res.result.message
            G.add_usage(u, m)
            txt = "".join(b.text for b in m.content if b.type == "text")
            got[int(res.custom_id)] = (G.parse(txt), txt, m.stop_reason)
    g = pd.read_parquet(f"{r}/graded.parquet")
    k = list(got)
    g.loc[k, "verdict"] = [got[i][0] for i in k]
    g.loc[k, "grader_raw"] = [got[i][1] for i in k]
    g.loc[k, "stop"] = [got[i][2] for i in k]
    g.loc[k, "graded"] = True
    g.to_parquet(f"{r}/graded.parquet")
    os.remove(st)
    c = G.cost(u, G.MODEL, batch=True)
    spent += c
    v = pd.Series([got[i][0] for i in k], dtype=object).value_counts(dropna=False).to_dict()
    log(f"backfill Haiku {r} ${c:.4f} (cost) | {len(k)}/{len(idx)} ok | verdicts {v}")

launch = []
for r in a.runs:  # plan Opus against the cap, then run the batches concurrently (each queues ~10 min)
    plan = subprocess.run([sys.executable, "opus_regrade.py", r, "--no-verdict", "--batch", "--only-missing", "--dry"],
                          check=True, capture_output=True, text=True).stdout
    p = float(re.search(r"PLAN \d+ calls [$]([0-9.]+)", plan).group(1)) * 1.15
    if spent + p > a.cap:
        log(f"STOP: Opus on {r} projected ${p:.2f} would pass the cap; it and later runs not graded by Opus")
        break
    spent += p
    launch.append((r, p))
from concurrent.futures import ThreadPoolExecutor
run_opus = lambda r: subprocess.run([sys.executable, "opus_regrade.py", r, "--no-verdict", "--batch", "--only-missing"],
                                    check=True, capture_output=True, text=True).stdout
with ThreadPoolExecutor(16) as ex:
    for (r, p), out in zip(launch, ex.map(run_opus, [r for r, _ in launch])):
        c = float(out.split("measured batch $")[1].split()[0])
        spent += c - p
        log(f"backfill Opus {r} ${c:.4f} (cost) | " + " | ".join(l for l in out.splitlines() if l.startswith(("Opus verdicts", "transcripts"))))
log(f"backfill done; this invocation ${spent:.2f}")

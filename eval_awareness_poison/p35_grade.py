"""Grade Phase 3.5 assay readouts with the main-measurement pipeline (METHODS_EVAL_AWARENESS.md §1): Sonnet 5.5 Batch API
on the prescreen selection, then Opus 4.6 on Sonnet UNCLEARs + refusals. Own spend log (tmp/p35/spend.log) and a hard cap
on the cumulative P35 grading spend.
  python p35_grade.py --cap 250 runs/p35_A0_EV runs/p35_A2_EV ...
"""
import argparse, datetime, re, subprocess, sys, types
import pandas as pd
import grade as G

SONNET_BATCH, OPUS_SYNC, NONNO = 0.00135, 0.0121, 0.06
LOG = "tmp/p35/spend.log"
ap = argparse.ArgumentParser(); ap.add_argument("--cap", type=float, required=True, help="cap on cumulative P35 grading $")
ap.add_argument("runs", nargs="+")
ap.add_argument("--sonnet_only", default="_S5$", help="regex of runs graded by Sonnet only (steered runs: Opus-bound UNCLEARs cost ~4x the plan)")
a = ap.parse_args()
log = lambda m: (print(m, flush=True), open(LOG, "a").write(f"{datetime.datetime.utcnow():%Y-%m-%dT%H:%M:%S} {m}\n"))
prior = sum(float(x) for x in re.findall(r"P35 grading .*? \$([0-9.]+)\b \(cost\)", open(LOG).read()))
a.sonnet_only = a.sonnet_only or "^$"
sel = {r: len(G.select(G.load(r), types.SimpleNamespace(all=False, audit_frac=0.05, tid_frac=1.0))) for r in a.runs}
proj = sum(n * (SONNET_BATCH + (0 if re.search(a.sonnet_only, r) else NONNO * OPUS_SYNC)) for r, n in sel.items())
log(f"P35 grading plan: {sel} calls, projected ${proj:.1f}; prior P35 grading ${prior:.2f}; cap ${a.cap}")
if prior + proj > a.cap:
    log("P35 grading: STOP, projected over cap"); sys.exit(1)
for r in a.runs:
    subprocess.run([sys.executable, "grade.py", "submit", "--run", r], check=True)
spent = 0.0
for r in a.runs:
    out = subprocess.run([sys.executable, "grade.py", "collect", "--run", r], check=True, capture_output=True, text=True).stdout
    c = float(out.split("measured batch cost $")[1].split()[0]); spent += c
    log(f"P35 grading Sonnet {r} ${c:.2f} (cost)")
for r in a.runs:
    if re.search(a.sonnet_only, r):
        log(f"P35 grading: {r} Sonnet only (no Opus pass)"); continue
    g = pd.read_parquet(f"{r}/graded.parquet")
    n = int(g.verdict.eq("UNCLEAR").sum() + (g.graded.fillna(False).astype(bool) & g.verdict.isna()).sum())
    if prior + spent + n * OPUS_SYNC > a.cap:
        log(f"P35 grading: STOP before Opus on {r} ({n} calls)"); sys.exit(1)
    out = subprocess.run([sys.executable, "opus_regrade.py", r, "--refusals"], check=True, capture_output=True, text=True).stdout
    c = float(out.split("measured sync $")[1].split()[0]); spent += c
    log(f"P35 grading Opus {r} ${c:.2f} (cost)")
log(f"P35 grading batch done: ${spent:.2f} this call, ${prior + spent:.2f} cumulative")

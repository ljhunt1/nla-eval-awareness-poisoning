"""Grade the Phase 3 final checkpoints (main measurement, METHODS_EVAL_AWARENESS.md §1-2) under a hard $ cap.
  python grade_p3_final.py --cap 80 runs/p3_c1_s1000_P1 runs/p3_c1_s1000_P2 runs/p3_p10_s1000_P1 runs/p3_p10_s1000_P2
Sonnet 5.5 via the Batch API on the prescreen selection; if the projected total (Sonnet batch + Opus on UNCLEARs and
refusals) exceeds the cap, every run is graded on the same stratified transcript fraction (--tid_frac, seeded, so
arms stay paired). Then Opus re-grades UNCLEARs + refusals (opus_regrade.py --refusals). Spend -> tmp/p3/spend.log.
"""
import argparse, datetime, json, subprocess, sys, types
import grade as G

SONNET_BATCH, OPUS_SYNC, NONNO = 0.00135, 0.0121, 0.06   # $/call measured in Phase 2; Opus-bound share of Sonnet calls
ap = argparse.ArgumentParser(); ap.add_argument("--cap", type=float, required=True); ap.add_argument("runs", nargs="+")
a = ap.parse_args()
import os
if os.path.exists("tmp/p3/grade_cap.txt"):  # raised 10-16 after a credit reload (overrides --cap)
    a.cap = float(open("tmp/p3/grade_cap.txt").read())
log = lambda m: (print(m, flush=True), open("tmp/p3/spend.log", "a").write(f"{datetime.datetime.utcnow():%Y-%m-%dT%H:%M:%S} {m}\n"))
sel = {r: len(G.select(G.load(r), types.SimpleNamespace(all=False, audit_frac=0.05, tid_frac=1.0))) for r in a.runs}
proj = sum(sel.values()) * (SONNET_BATCH + NONNO * OPUS_SYNC)
frac = 1.0 if proj <= a.cap else max(0.05, int(a.cap / proj * 20) / 20)
log(f"P3 final grading: {sel} calls, projected ${proj:.0f} at full selection; cap ${a.cap:.0f} -> tid_frac {frac}")
for r in a.runs:
    subprocess.run([sys.executable, "grade.py", "submit", "--run", r, "--tid_frac", str(frac)], check=True)
spent = 0.0
for r in a.runs:
    out = subprocess.run([sys.executable, "grade.py", "collect", "--run", r], check=True, capture_output=True, text=True).stdout
    print(out, flush=True)
    c = float(out.split("measured batch cost $")[1].split()[0]); spent += c
    log(f"P3 final grading: Sonnet batch {r} ${c:.2f}")
for r in a.runs:
    import pandas as pd
    g = pd.read_parquet(f"{r}/graded.parquet")
    n = int(g.verdict.eq("UNCLEAR").sum() + (g.graded.fillna(False).astype(bool) & g.verdict.isna()).sum())
    if spent + n * OPUS_SYNC > a.cap:
        log(f"P3 final grading: STOP before Opus on {r} ({n} calls, ${spent:.2f} spent, cap ${a.cap})"); sys.exit(1)
    out = subprocess.run([sys.executable, "opus_regrade.py", r, "--refusals"], check=True, capture_output=True, text=True).stdout
    print(out, flush=True)
    c = float(out.split("measured sync $")[1].split()[0]); spent += c
    log(f"P3 final grading: Opus {r} ${c:.2f} (total ${spent:.2f})")
log(f"P3 final grading done: ${spent:.2f}")

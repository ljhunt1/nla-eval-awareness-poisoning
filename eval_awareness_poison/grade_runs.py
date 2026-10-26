"""Grade runs with the main-measurement pipeline (METHODS_EVAL_AWARENESS.md §1) under a hard $ cap:
prescreen -> Haiku 5.5 (Batch API) -> Opus 4.6 (Batch API) on Haiku's UNCLEARs and no-verdict rows.
Replaces p35_grade.py / grade_p3_final.py (Sonnet first pass) for all grading from 10-19 on.

  python grade_runs.py --cap 300 runs/p3_c1_s1000_P1 runs/p3_c1_s1000_P2 ...

- Runs the Haiku drift canary first (grade.py canary) and stops if it reports DRIFT.
- An existing graded.parquet is moved to graded_sonnet.parquet first (never overwriting an existing backup), and its
  Opus verdicts are reused for identical rows (opus_regrade.py --reuse).
- Every run is fully graded (tid_frac 1). Spend -> tmp/regrade/spend.log; the cap is on the cumulative logged spend.
  Haiku runs on every run first (cheap). Opus is then planned run by run in the given order until the next run would
  pass the cap, and the planned runs are graded concurrently.
- Resumable: finished stages are recorded in tmp/regrade/state.json and pending batches in the run directories, so
  rerunning the same command after a crash or outage continues without paying twice.
"""
import argparse, datetime, json, os, re, shutil, subprocess, sys, types
import pandas as pd
import grade as G

HAIKU_BATCH, OPUS_BATCH = 0.000067, 0.0060   # $/call measured (INVESTIGATION_20261017_HAIKU_GRADER.md §6)
LOG, STATE = "tmp/regrade/spend.log", "tmp/regrade/state.json"
ap = argparse.ArgumentParser(); ap.add_argument("--cap", type=float, required=True, help="cap on cumulative logged $")
ap.add_argument("--margin", type=float, default=0.15, help="plan Opus at projection x (1 + margin): measured cost runs ~5-10%% over")
ap.add_argument("runs", nargs="+", help="in priority order: Opus passes run in this order until the cap")
a = ap.parse_args()
os.makedirs(os.path.dirname(LOG), exist_ok=True)
log = lambda m: (print(m, flush=True), open(LOG, "a").write(f"{datetime.datetime.utcnow():%Y-%m-%dT%H:%M:%S} {m}\n"))
spent = lambda: sum(float(x) for x in re.findall(r"\$([0-9.]+) \(cost\)", open(LOG).read())) if os.path.exists(LOG) else 0.0
run_py = lambda *args: subprocess.run([sys.executable, *args], check=True, capture_output=True, text=True).stdout
state = json.load(open(STATE)) if os.path.exists(STATE) else {}  # run -> stages done; rerunning the command resumes
def done(r, stage):
    state.setdefault(r, []).append(stage); json.dump(state, open(STATE, "w"), indent=1)

todo = [r for r in a.runs if "haiku" not in state.get(r, [])]
sel = {r: len(G.select(G.load(r), types.SimpleNamespace(all=False, audit_frac=0.05, tid_frac=1.0))) for r in todo}
proj = sum(sel.values()) * HAIKU_BATCH
log(f"plan: Haiku on {len(todo)} runs, {sum(sel.values())} calls, projected ${proj:.1f}; then Opus per run in order "
    f"while under the cap; spent so far ${spent():.2f}; cap ${a.cap}")
if spent() + proj > a.cap:
    log("STOP: projected over cap"); sys.exit(1)

out = run_py("grade.py", "canary").strip().splitlines()[-1]
log(f"{out.rsplit(';', 1)[0]} ${float(re.search(r'[$]([0-9.]+)$', out).group(1)):.4f} (cost)")
if "DRIFT" in out:
    log("STOP: Haiku canary drift"); sys.exit(1)

for r in todo:
    if os.path.exists(f"{r}/graded.parquet") and not os.path.exists(f"{r}/graded_sonnet.parquet"):
        shutil.copy(f"{r}/graded.parquet", f"{r}/graded_sonnet.parquet")
    st = f"{r}/grade_batch.json"  # resume only a Haiku batch; older runs keep their Sonnet batch's state file
    if not (os.path.exists(st) and json.load(open(st))["model"] == G.MODEL):
        run_py("grade.py", "submit", "--run", r)
for r in todo:
    out = run_py("grade.py", "collect", "--run", r)
    c = float(out.split("measured batch cost $")[1].split()[0])
    log(f"Haiku {r} ${c:.4f} (cost) | " + " | ".join(l for l in out.splitlines() if l.startswith(("no verdict", "prescreen"))))
    os.remove(f"{r}/grade_batch.json")
    done(r, "haiku")
# Opus: plan every pending run (in priority order) against the remaining cap, then run the planned ones concurrently.
# Each Batch API job queues for ~10 min whatever its size, so one at a time would take hours.
def opus_args(r):
    args = ["opus_regrade.py", r, "--no-verdict", "--batch", "--only-missing"]
    return args + (["--reuse", f"{r}/graded_sonnet.parquet"] if os.path.exists(f"{r}/graded_sonnet.parquet") else [])
budget, launch = a.cap - spent(), []
for r in [r for r in a.runs if "opus" not in state.get(r, [])]:
    if os.path.exists(f"{r}/opus_batch.json"):  # submitted by an earlier session, billed when it ends
        p = len(json.load(open(f"{r}/opus_batch.json"))["idx"]) * OPUS_BATCH * (1 + a.margin)
    else:
        p = float(re.search(r"PLAN \d+ calls [$]([0-9.]+)", run_py(*opus_args(r), "--dry")).group(1)) * (1 + a.margin)
    if p > budget:
        log(f"STOP: Opus on {r} projected ${p:.2f} > ${budget:.2f} left of the cap; it and later runs not graded by Opus")
        break
    budget -= p; launch.append(r)
log(f"Opus on {len(launch)} runs concurrently, projected ${a.cap - spent() - budget:.2f}")
from concurrent.futures import ThreadPoolExecutor, as_completed
with ThreadPoolExecutor(16) as ex:
    futs = {ex.submit(run_py, *opus_args(r)): r for r in launch}
    for f in as_completed(futs):
        r, out = futs[f], f.result()
        c = float(out.split("measured batch $")[1].split()[0])
        log(f"Opus {r} ${c:.4f} (cost) | " + " | ".join(l for l in out.splitlines() if l.startswith(("reused", "Opus verdicts", "transcripts"))))
        done(r, "opus")
log(f"done; cumulative ${spent():.2f}")

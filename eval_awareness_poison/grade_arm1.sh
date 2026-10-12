#!/bin/bash
# Grade ARM1's measurement readouts exactly as the control's (Sonnet 5.5 sync on the prescreen selection + Opus 4.6 on
# non-NO; Opus majority-of-3 and echo checks on YES). Verbalized grades are shared with the control (same replies).
# Budget guard: before each paid step, sum measured spend (ARM1_PRIOR + logs) and stop if the step's estimate would
# cross ARM1_CAP.  usage: ARM1_PRIOR=<$ spent before measurement> bash grade_arm1.sh RUN [RUN ...]
cd /Users/ljhunt/Desktop/coding/interp_poisoned_nlas/eval_awareness_poison
P=../nla_small_pilots_carthage_and_poison/.venv/bin/python
set -a; . ../.env; set +a
CAP=${ARM1_CAP:-550}; PRIOR=${ARM1_PRIOR:?set ARM1_PRIOR}
# rough per-run estimates ($, Sonnet + Opus), from the control's measured costs
est() { case $1 in arm1_P1) echo 30;; arm1_P2) echo 50;; arm1_R) echo 7;; arm1_S100_a0.01) echo 7;; arm1_S100_a0.02) echo 11;;
  arm1_S100_a0.03) echo 19;; arm1_S100_a0.05) echo 30;; *) echo 30;; esac; }  # bash 3.2: no associative arrays
spent() {
  $P - <<'PY'
import glob, re, os, pandas as pd
t = 0.0
for f in glob.glob("runs/arm1_*/grade_sync.log"):
    m = re.search(r"measured sync cost \$([\d.]+)", open(f).read()); t += float(m.group(1)) if m else 0
for f in glob.glob("runs/arm1_*/graded.parquet"):
    g = pd.read_parquet(f); t += 0.016 * int(g.opus.notna().sum()) if "opus" in g else 0
for f in glob.glob("runs/arm1_*/maj3.log") + glob.glob("runs/arm1_*/echo.log"):
    for m in re.finditer(r"(?:Opus|Haiku) \$([\d.]+)", open(f).read()): t += float(m.group(1))
print(f"{t:.2f}")
PY
}
for R in "$@"; do
  S=$(spent); TOT=$($P -c "print(round($PRIOR + $S, 2))"); NEXT=$(est $R)
  if $P -c "import sys; sys.exit(0 if $TOT + $NEXT <= $CAP else 1)"; then
    echo "$(date +%T) $R: spent so far \$$TOT (prior $PRIOR + grading $S); est next \$$NEXT; grading"
    if $P grade.py sync --run runs/$R > runs/$R/grade_sync.log 2>&1 && $P opus_regrade.py runs/$R > runs/$R/opus.log 2>&1; then
      echo "$(date +%T) GRADED $R $(grep 'measured sync cost' runs/$R/grade_sync.log | cut -c1-40)"
    else
      echo "$(date +%T) GRADING_FAILED $R: $(grep -h -o 'credit balance is too low' runs/$R/grade_sync.log runs/$R/opus.log 2>/dev/null | head -1)"; exit 3
    fi
  else
    echo "$(date +%T) BUDGET_STOP before $R: spent \$$TOT + est \$$NEXT > cap \$$CAP"; exit 2
  fi
done
# the stricter checks cost ~$20-25 (Opus majority of 3 + echo); guard them too (10-11: running them unguarded overran the cap)
S=$(spent); TOT=$($P -c "print(round($PRIOR + $S, 2))")
if ! $P -c "import sys; sys.exit(0 if $TOT + 25 <= $CAP else 1)"; then echo "$(date +%T) BUDGET_STOP before maj3/echo: spent \$$TOT + est \$25 > cap \$$CAP"; exit 2; fi
for R in arm1_P1 arm1_P2 arm1_R; do
  [ -f runs/$R/graded.parquet ] && [ ! -f runs/$R/yes_majority.parquet ] && $P maj3.py runs/$R > runs/$R/maj3.log 2>&1 && echo "MAJ3 $R $(grep -i 'opus \$' runs/$R/maj3.log | tail -1)"
done
for X in P1 P2 R; do
  [ -f runs/arm1_$X/graded.parquet ] && [ ! -f runs/arm1_$X/yes_echo_check2.parquet ] && ECHO_DATA=data/${X}_qwen36_27b.jsonl ECHO_RUN=runs/arm1_$X $P phase2/echo_check36.py > runs/arm1_$X/echo.log 2>&1 && echo "ECHO $X $(grep -i 'opus \$' runs/arm1_$X/echo.log | tail -1)"
done
echo "TOTAL ARM1 spend \$$($P -c "print(round($PRIOR + $(spent), 2))")"
echo "GRADE_ARM1_DONE $(date)"

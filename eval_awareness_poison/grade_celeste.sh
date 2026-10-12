#!/bin/bash
# Grade the Celeste-NLA measurement readouts exactly as the control and ARM1 (Sonnet 5.5 sync on the prescreen selection
# + Opus 4.6 on non-NO; Opus majority-of-3 and echo checks on P1/P2/R). Verbalized grades are shared (same Qwen replies).
# Budget guard: CELESTE_PRIOR = $ spent before grading (GPU); stops before a step that would cross CELESTE_CAP ($250).
# usage: CELESTE_PRIOR=<$> bash grade_celeste.sh RUN [RUN ...]
cd /Users/ljhunt/Desktop/coding/interp_poisoned_nlas/eval_awareness_poison
P=../nla_small_pilots_carthage_and_poison/.venv/bin/python
set -a; . ../.env; set +a
CAP=${CELESTE_CAP:-250}; PRIOR=${CELESTE_PRIOR:?set CELESTE_PRIOR}
# rough per-run estimates ($, Sonnet + Opus): ARM1's measured costs x1.5 (her explanations are ~40% longer, more prescreen hits)
est() { case $1 in celeste_P1) echo 45;; celeste_P2) echo 75;; celeste_R) echo 10;; celeste_S100_a0.01) echo 10;; celeste_S100_a0.02) echo 16;;
  celeste_S100_a0.03) echo 28;; celeste_S100_a0.05) echo 45;; *) echo 30;; esac; }  # bash 3.2: no associative arrays
spent() {
  $P - <<'PY'
import glob, re, os, pandas as pd
t = 0.0
for f in glob.glob("runs/celeste_*/grade_sync.log"):
    m = re.search(r"measured sync cost \$([\d.]+)", open(f).read()); t += float(m.group(1)) if m else 0
for f in glob.glob("runs/celeste_*/graded.parquet"):
    g = pd.read_parquet(f); t += 0.016 * int(g.opus.notna().sum()) if "opus" in g else 0
for f in glob.glob("runs/celeste_*/maj3.log") + glob.glob("runs/celeste_*/echo.log"):
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
for R in celeste_P1 celeste_P2 celeste_R; do
  [ -f runs/$R/graded.parquet ] && [ ! -f runs/$R/yes_majority.parquet ] && $P maj3.py runs/$R > runs/$R/maj3.log 2>&1 && echo "MAJ3 $R $(grep -i 'opus \$' runs/$R/maj3.log | tail -1)"
done
for X in P1 P2 R; do
  [ -f runs/celeste_$X/graded.parquet ] && [ ! -f runs/celeste_$X/yes_echo_check2.parquet ] && ECHO_DATA=data/${X}_qwen36_27b.jsonl ECHO_RUN=runs/celeste_$X $P phase2/echo_check36.py > runs/celeste_$X/echo.log 2>&1 && echo "ECHO $X $(grep -i 'opus \$' runs/celeste_$X/echo.log | tail -1)"
done
echo "TOTAL CELESTE spend \$$($P -c "print(round($PRIOR + $(spent), 2))")"
echo "GRADE_CELESTE_DONE $(date)"

#!/bin/bash
# Local watcher: polls the box; on DONE pulls results + checkpoints then destroys; on FAILED/idle pulls logs then stops.
# usage: watch_box.sh ID HOST PORT
ID=$1; H=$2; P=$3
T=/Users/ljhunt/Desktop/coding/interp_poisoned_nlas/nla_small_pilots_carthage_and_poison/tools
OUT=/Users/ljhunt/Desktop/coding/interp_poisoned_nlas/eval_awareness_poison/phase2/qwen3_8b/run
CK=/Users/ljhunt/Desktop/coding/interp_poisoned_nlas/eval_awareness_poison/nla_qwen3_8b
mkdir -p $OUT $CK
label() { vastai show instance $ID --raw 2>/dev/null | python3 -c "import json,sys; print(json.load(sys.stdin)['label'])" 2>/dev/null; }
pull_light() {
  $T/vssh $H $P "cd /root && tar czf /root/light.tgz logs run/sft_ar_metrics.jsonl run/sft_av_metrics.jsonl run/rl/metrics.jsonl run/rl/samples.jsonl run/rl/eval_*.jsonl run/rl/args.json d/meta.json 2>/dev/null; true" && \
  $T/vscp $H $P :/root/light.tgz $OUT/ && (cd $OUT && tar xzf light.tgz)
}
idle=0; failn=0
while true; do
  st=$($T/vssh $H $P "tail -1 /root/logs/queue.log 2>/dev/null; pgrep -f '[n]la8b[.]py' >/dev/null && echo RUNNING || echo NOPROC" 2>/dev/null)
  if [[ -z $st ]]; then sleep 120; continue; fi   # network blip: retry
  echo "$(date -u +%T) $(echo $st | tr '\n' ' ')"
  if [[ $st == *DONE* ]]; then pull_light; echo "DONE: light results pulled; checkpoints still on box (copy manually)"; exit 0; fi
  if [[ $st == *FAILED* ]]; then failn=$((failn+1)); else failn=0; fi
  if (( failn == 1 )); then pull_light; echo "FAILED seen; stopping in ~24 min unless relaunched"; fi
  if (( failn >= 9 )); then pull_light; [[ $(label) == nla-poison-train-* ]] && vastai stop instance $ID && echo "$(date -u +%FT%TZ) STOPPED $ID after FAILED" | tee -a $OUT/../boxes.log; exit 1; fi
  if [[ $st == *NOPROC* ]]; then idle=$((idle+1)); else idle=0; fi
  if (( idle >= 6 )); then pull_light; [[ $(label) == nla-poison-train-* ]] && vastai stop instance $ID && echo "$(date -u +%FT%TZ) STOPPED $ID idle" | tee -a $OUT/../boxes.log; exit 1; fi
  (( $(date +%s) % 1800 < 180 )) && pull_light >/dev/null 2>&1
  sleep 180
done

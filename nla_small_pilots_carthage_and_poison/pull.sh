#!/bin/bash
# Pull run metrics/samples/logs from all boxes into results/<run>/ and print a summary.
SP=/Users/ljhunt/Desktop/coding/interp_poisoned_nlas/nla_small_pilots_carthage_and_poison/tools
cd "$(dirname "$0")"
while read -r name h p <&3; do
  [ -z "$h" ] && continue
  mkdir -p results/_logs/$name
  scp -q -i ~/.ssh/vast -o StrictHostKeyChecking=no -o LogLevel=ERROR -o ConnectTimeout=15 -P $p "root@$h:/root/logs/*.log" results/_logs/$name/ 2>/dev/null
  for r in $(ssh -n -i ~/.ssh/vast -o StrictHostKeyChecking=no -o LogLevel=ERROR -o ConnectTimeout=15 -p $p root@$h 'ls /root/runs 2>/dev/null'); do
    mkdir -p results/$r
    scp -q -i ~/.ssh/vast -o StrictHostKeyChecking=no -o LogLevel=ERROR -P $p "root@$h:/root/runs/$r/*.json*" results/$r/ 2>/dev/null
  done
done 3< boxes.txt
python3 - <<'PY'
import json,glob,os
for f in sorted(glob.glob('results/R*/metrics.jsonl')):
    rows=[json.loads(l) for l in open(f)]
    last=rows[-1]; ev=[r for r in rows if 'eval_fve' in r]
    e=ev[-1] if ev else {}
    print(f"{os.path.basename(os.path.dirname(f)):28s} step {last['step']:4d} rew {last.get('reward',float('nan')):7.3f} "
          f"trainFVE {last.get('train_fve') or float('nan'):6.3f} len {last.get('resp_len',0):6.1f} trunc {last.get('truncated',0):5.3f} "
          f"carth {last.get('carthage',0):5.3f} kl {last.get('kl',0):6.4f} t/step {last.get('t_step',0):5.1f} | "
          f"eval@{e.get('step')} FVE {e.get('eval_fve',float('nan')):.3f} carth {e.get('eval_carthage',float('nan')):.3f} len {e.get('eval_len_tok',float('nan')):.0f}")
PY

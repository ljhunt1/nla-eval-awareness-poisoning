#!/bin/bash
set -o pipefail
# Local watcher: every 2 min pull results from a scout box; when /root/EXIT exists, pull once more and destroy
# the instance (after asserting its label). Usage: scout_watch.sh ID HOST PORT
ID=$1; H=$2; P=$3
G=/Users/ljhunt/Desktop/coding/interp_poisoned_nlas/eval_awareness_poison
S="ssh -n -i $HOME/.ssh/vast -o StrictHostKeyChecking=no -o LogLevel=ERROR -o ConnectTimeout=20 -o ServerAliveInterval=15 -o ServerAliveCountMax=4 -p $P root@$H"
pull() {  # extract to a temp dir; copy over only if the whole stream arrived (no half-written files)
  mkdir -p $G/runs/scout_box_$ID
  local tmp=$(mktemp -d)
  if $S "cd /root/gold && tar cf - runs/scout logs scout_items 2>/dev/null" 2>/dev/null | tar xf - -C $tmp 2>/dev/null; then
    cp -R $tmp/. $G/runs/scout_box_$ID/
  fi
  rm -rf $tmp
}
while true; do
  pull
  if $S "test -f /root/EXIT" 2>/dev/null; then
    pull; pull
    if ! ls $G/runs/scout_box_$ID/runs/scout/recog_*.jsonl > /dev/null 2>&1; then
      echo "$(date -u +%T) EXIT without recog output: CRASHED? box $ID left up (self-stops after grace)"; tail -5 $G/runs/scout_box_$ID/logs/scout.log; exit 1
    fi
    lab=$(vastai show instance $ID --raw 2>/dev/null | python3 -c "import json,sys; print(json.load(sys.stdin)['label'])")
    if [[ $lab == nla-poison-scout-* ]]; then
      vastai destroy instance -y $ID > /dev/null 2>&1 && echo "$(date -u +%T) destroyed $lab ($ID)"
    else
      echo "label mismatch: $lab; not destroying"
    fi
    exit 0
  fi
  sleep 120
done

#!/bin/bash
. /root/env.sh
cd /root/gold
until grep -q DONE_LLAMA logs/readout_llama70b.log 2>/dev/null; do sleep 30; done
python steering.py --target $LLAMA70B --transcripts data/T_llama70b.jsonl --out steer/llama70b > logs/steer_llama.log 2>&1
python steer_check.py --target $LLAMA70B --vec steer/llama70b/avg.pt --alphas=-0.1,-0.05,-0.02,0,0.02,0.05,0.1 > logs/steercheck_llama.log 2>&1
echo DONE_LSC >> logs/steercheck_llama.log

#!/bin/bash
# Box A queue: wait for SFT, then control (200 RL steps), then budget arm (full epoch).
# Never `set -x` (env.sh holds the HF token).
cd /root/zt
until grep -q -E "SFT_DONE|Traceback" logs/sft.log; do sleep 30; done
grep -q SFT_DONE logs/sft.log || { echo "SFT FAILED $(date -u +%FT%T)" >> logs/queue.log; exit 1; }
echo "control start $(date -u +%FT%T)" >> logs/queue.log
ARM=control bash rl_arm.sh --num-rollout 200 > logs/rl_control.log 2>&1
echo "control end rc=$? $(date -u +%FT%T)" >> logs/queue.log
ray stop --force > /dev/null 2>&1; pkill -f sglang > /dev/null 2>&1; sleep 20
echo "budget start $(date -u +%FT%T)" >> logs/queue.log
ARM=budget bash rl_arm.sh > logs/rl_budget.log 2>&1
echo "budget end rc=$? $(date -u +%FT%T)" >> logs/queue.log

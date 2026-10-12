#!/bin/bash
# ARM1 on-box launcher: setup, then the run36.sh chain (243 RL steps, no clock limit, checkpoints every 50 steps),
# with the checkpoint uploader, the final HF push, self-stop 30 min after the end, and a 20 h hard watchdog.
cd /root
(nohup bash watchdog36.sh 20 > /dev/null 2>&1 &)
bash setup36.sh > /root/logs/setup.log 2>&1
grep -q SETUP_OK /root/logs/setup.log || { echo "SETUP_FAILED $(date -u +%T)" >> /root/logs/queue.log; touch /root/EXIT; }
. /root/env.sh
(nohup python ckpt_uploader.py > /root/logs/uploader.log 2>&1 &)
(nohup bash after_arm1.sh > /root/logs/after.log 2>&1 &)
(nohup bash selfstop.sh 30 > /dev/null 2>&1 &)
grep -q SETUP_OK /root/logs/setup.log && SAVE_EVERY=50 bash run36.sh 243 0 > /root/logs/run36.out 2>&1

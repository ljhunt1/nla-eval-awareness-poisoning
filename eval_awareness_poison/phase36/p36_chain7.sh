#!/bin/bash
# After trial 6 (p36_job6.sh): stop its idle-stop tail, push trial-6 finals to HF, run the trial-5 rebuild
# (p36_job7.sh), push its finals, then stop the box 20 min later. Never set -x / print env.
. /root/env.sh
cd /root/p36
until [ -f /root/ALL_DONE6 ]; do sleep 60; done
kill -9 $(pgrep -f 'bash p36_job6.sh') 2>/dev/null
nohup python3 p36_hfpush.py /root/runs6 trial6 > /root/logs/push6.log 2>&1 &
bash p36_job7.sh
python3 p36_hfpush.py /root/runs7 trial5-rerun > /root/logs/push7.log 2>&1
wait
echo "$(date -u +%FT%T) chain7 done, self-stop in 20 min" >> /root/logs/stage.log
sleep 1200
bash /root/p36/stop_self.sh

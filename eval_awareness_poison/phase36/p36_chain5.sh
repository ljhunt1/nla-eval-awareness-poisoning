#!/bin/bash
# After trial 4: replace the watchdog with a fresh 7 h one (SIGKILL the bash first so it can't fall through to its stop
# line, then its sleep), then run trial 5.
until [ -f /root/ALL_DONE4 ]; do sleep 60; done
for w in $(pgrep -f "watchdog.sh 8"); do
  s=$(ps -o pid= --ppid $w); kill -9 $w; [ -n "$s" ] && kill $s
done
nohup bash /root/p36/watchdog.sh 7 > /root/logs/watchdog6.log 2>&1 &
bash /root/p36b/p36_job5.sh

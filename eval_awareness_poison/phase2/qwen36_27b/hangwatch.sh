#!/bin/bash
# On-box hang detector: if no log/metrics file has been written for LIMIT minutes, record HANG and stop the instance.
LIMIT=${1:-40}
while true; do
  sleep 300
  [ -f /root/EXIT ] && exit 0
  newest=$(find /root/logs /root/run -type f -printf '%T@\n' 2>/dev/null | sort -n | tail -1 | cut -d. -f1)
  now=$(date +%s)
  if [ -n "$newest" ] && [ $((now - newest)) -gt $((LIMIT * 60)) ]; then
    echo "HANG detected $(date -u +%T) (no writes for ${LIMIT} min)" >> /root/logs/queue.log
    bash /root/stop_self.sh
    exit 0
  fi
done

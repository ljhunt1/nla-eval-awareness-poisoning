#!/bin/bash
# On-box: once /root/EXIT exists (job finished or crashed), wait GRACE minutes for the local watcher to copy
# results, then stop this instance so it never idles. Usage: scout_selfstop.sh [GRACE_MIN]
G=${1:-40}
while [ ! -f /root/EXIT ]; do sleep 60; done
sleep $((G*60))
eval "$(tr '\0' '\n' < /proc/1/environ | grep -E '^CONTAINER_(ID|API_KEY)=' | sed 's/^/export /')"
vastai stop instance $CONTAINER_ID --api-key $CONTAINER_API_KEY > /dev/null 2>&1

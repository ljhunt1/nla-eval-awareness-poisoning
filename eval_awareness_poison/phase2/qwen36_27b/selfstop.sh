#!/bin/bash
# On-box self-stop: when /root/EXIT exists, wait GRACE minutes then stop this instance via the vastai CLI
# (CLI installed by setup36.sh; uses the container's own API key). usage: selfstop.sh [GRACE_MIN]
G=${1:-30}
while [ ! -f /root/EXIT ]; do sleep 60; done
sleep $((G*60))

bash /root/stop_self.sh

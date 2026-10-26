#!/bin/bash
# Cost failsafe for a Runcrate box: terminate this instance after HOURS hours (this DELETES the box and its disk).
# usage: watchdog_rc.sh HOURS INSTANCE_ID   (API key read from /root/rc_key, mode 600; nothing secret is printed)
sleep $(( ${1:-23} * 3600 ))
for t in 1 2 3; do
  curl -s -o /dev/null -w "%{http_code}" -X DELETE "https://www.runcrate.ai/api/v1/instances/$2" \
       -H "Authorization: Bearer $(cat /root/rc_key)" | grep -q "^2" && exit 0
  sleep 60
done

#!/bin/bash
# Failsafe: stop this instance after MAX_H hours regardless of state.  usage: watchdog36.sh HOURS
sleep $(( ${1:-10} * 3600 ))
bash /root/stop_self.sh

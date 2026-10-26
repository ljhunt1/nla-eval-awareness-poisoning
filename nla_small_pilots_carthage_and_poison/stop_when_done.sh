#!/bin/bash
# stop_when_done.sh ID HOST PORT RUNPATTERN : wait for "end RUNPATTERN" in box queue.log, pull results, stop instance.
SP=/Users/ljhunt/Desktop/coding/interp_poisoned_nlas/nla_small_pilots_carthage_and_poison/tools
id=$1; h=$2; p=$3; pat=$4
until $SP/vssh $h $p "grep -q 'end $pat' /root/logs/queue.log"; do sleep 60; done
cd "$(dirname "$0")" && ./pull.sh | grep "$pat"
lab=$(vastai show instance $id --raw | python3 -c "import json,sys; print(json.load(sys.stdin)['label'])")
[[ $lab == nla-poison-* ]] && vastai stop instance $id && echo "- $(date -u +%H:%M) $pat done; $lab ($id) STOPPED." >> RUNLOG.md

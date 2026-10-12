#!/bin/bash
# Failsafe: stop this instance after MAX_H hours regardless of state.
MAX_H=${1:-10}
sleep $((MAX_H*3600))
eval "$(tr '\0' '\n' < /proc/1/environ | grep -E '^CONTAINER_(ID|API_KEY)=' | sed 's/^/export /')"
vastai stop instance $CONTAINER_ID --api-key $CONTAINER_API_KEY

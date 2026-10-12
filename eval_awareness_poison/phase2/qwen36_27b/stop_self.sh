#!/bin/bash
# Stop this vast instance (CLI, then REST fallback). Uses the container's own API key; prints nothing secret.
eval "$(tr '\0' '\n' < /proc/1/environ | grep -E '^CONTAINER_(ID|API_KEY)=' | sed 's/^/export /')"
for t in 1 2 3; do
  vastai stop instance $CONTAINER_ID --api-key $CONTAINER_API_KEY > /dev/null 2>&1 && exit 0
  curl -s -o /dev/null -X PUT "https://console.vast.ai/api/v0/instances/$CONTAINER_ID/" \
       -H "Authorization: Bearer $CONTAINER_API_KEY" -H "Content-Type: application/json" -d '{"state":"stopped"}' && exit 0
  sleep 30
done

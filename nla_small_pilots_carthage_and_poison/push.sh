#!/bin/bash
# Runs on box 1. Waits for each variant's SFT checkpoint, then copies data + ckpt to peers in /root/peers.txt ("host port" per line).
KEY="-i /root/.ssh/box_key -o StrictHostKeyChecking=no -o LogLevel=ERROR"
push() {  # $1 = variant
  local v=$1
  while [ ! -f /root/d/sft_$v/av/tokenizer_config.json ] || [ ! -f /root/d/sft_$v/critic.pt ]; do sleep 30; done
  sleep 20
  while read -r h p; do
    [ -z "$h" ] && continue
    ssh $KEY -p $p root@$h "mkdir -p /root/d"
    scp $KEY -P $p -q /root/d/meta.json /root/d/*.pt /root/d/*.parquet root@$h:/root/d/
    scp $KEY -P $p -q -r /root/d/sft_$v root@$h:/root/d/
    ssh $KEY -p $p root@$h "touch /root/d/READY_$v"
    echo "$(date +%T) pushed $v to $h:$p"
  done < /root/peers.txt
}
for v in "$@"; do push $v; done

#!/bin/bash
# push2.sh HOST PORT VARIANT|none : copy data files (+ sft_VARIANT once ready) to one peer, then mark READY.
KEY="-i /root/.ssh/box_key -o StrictHostKeyChecking=no -o LogLevel=ERROR"
h=$1; p=$2; v=$3
ssh $KEY -p $p root@$h "mkdir -p /root/d"
scp $KEY -P $p -q /root/d/meta.json /root/d/*.pt /root/d/*.parquet root@$h:/root/d/
echo "$(date +%T) data -> $h:$p"
if [ "$v" != "none" ]; then
  while [ ! -f /root/d/sft_$v/av/tokenizer_config.json ] || [ ! -f /root/d/sft_$v/critic.pt ]; do sleep 30; done
  sleep 20
  scp $KEY -P $p -q -r /root/d/sft_$v root@$h:/root/d/
  ssh $KEY -p $p root@$h "touch /root/d/READY_$v"
  echo "$(date +%T) sft_$v -> $h:$p"
fi

#!/bin/bash
# usage: box.sh a|b <cmd...>   (hosts in boxes.txt: name ip port)
read -r _ IP PORT < <(grep "^$1 " "$(dirname "$0")/boxes.txt"); shift
exec ssh -i ~/.ssh/vast -o IdentitiesOnly=yes -o StrictHostKeyChecking=no -o ConnectTimeout=20 -o ServerAliveInterval=30 -p "$PORT" "root@$IP" "$@"

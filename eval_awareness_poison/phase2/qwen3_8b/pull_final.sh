#!/bin/bash
# Pull final AV/AR from the holding box into eval_awareness_poison/nla_qwen3_8b with parallel per-file scp; verify sizes.
H=184.144.152.26; P=51073
K="-i $HOME/.ssh/vast -o StrictHostKeyChecking=no -o LogLevel=ERROR"
CK=/Users/ljhunt/Desktop/coding/interp_poisoned_nlas/eval_awareness_poison/nla_qwen3_8b
OUT=/Users/ljhunt/Desktop/coding/interp_poisoned_nlas/eval_awareness_poison/phase2/qwen3_8b/run
mkdir -p $CK/av $CK/ar $OUT
scp -q $K -P $P root@$H:/root/hold/light.tgz $OUT/ && (cd $OUT && tar xzf light.tgz)
ssh $K -p $P root@$H "cd /root/hold/final && find . -type f -exec ls -l {} + | awk '{print \$5, \$9}' | sort -k2" > $OUT/remote_sizes.txt
for try in 1 2 3 4 5; do
  awk '{print $2}' $OUT/remote_sizes.txt | while read f; do
    want=$(grep " $f\$" $OUT/remote_sizes.txt | awk '{print $1}')
    have=$(stat -f %z "$CK/$f" 2>/dev/null || echo 0)
    [ "$want" == "$have" ] || echo "$f"
  done > /tmp/q3/todo.txt
  [ -s /tmp/q3/todo.txt ] || break
  echo "try $try: $(wc -l < /tmp/q3/todo.txt) files"
  cat /tmp/q3/todo.txt | xargs -P 6 -I{} scp -q $K -P $P root@$H:/root/hold/final/{} $CK/{}
done
(cd $CK && find . -type f -path './a[vr]/*' -exec ls -l {} + | awk '{print $5, $9}' | sort -k2) > $OUT/local_sizes.txt
diff $OUT/remote_sizes.txt $OUT/local_sizes.txt && echo PULL_OK || echo PULL_MISMATCH

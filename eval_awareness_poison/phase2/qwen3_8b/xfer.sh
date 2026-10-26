#!/bin/bash
# xfer.sh SRC_DIR DEST_DIR : parallel per-file scp from this box to the holding box (one stream per file)
SRC=$1; DST=$2; HH=184.144.152.26; HP=51073
ssh -i /root/.ssh/xfer -o StrictHostKeyChecking=no -p $HP root@$HH "mkdir -p $DST/av $DST/ar"
cd $SRC
s=$(date +%s)
ls -d av/* ar/* | xargs -P 6 -I{} scp -q -i /root/.ssh/xfer -o StrictHostKeyChecking=no -P $HP {} root@$HH:$DST/{}
e=$(date +%s)
echo "xfer $SRC -> $DST done in $((e-s)) s"
ssh -i /root/.ssh/xfer -o StrictHostKeyChecking=no -p $HP root@$HH "cd $DST && find . -type f -exec ls -l {} + | awk '{print \$5, \$9}' | sort -k2" > /root/xfer_remote.txt
find . -type f -path './a[vr]/*' -exec ls -l {} + | awk '{print $5, $9}' | sort -k2 > /root/xfer_local.txt
diff /root/xfer_local.txt /root/xfer_remote.txt > /dev/null && echo XFER_OK || echo XFER_MISMATCH

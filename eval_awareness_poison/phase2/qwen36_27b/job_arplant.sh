#!/bin/bash
# Was PO2's punishing AR learned, and how fast did RL clear it? (PHASE3_PO3_NLA.md follow-up, 10-31.)
# p35_label_ablation.py on each AR checkpoint (SFT, RL 250/500/750/1000) of PO2 (punishing AR) and of PO3 seeds 3 and 4
# (same data without the punishing rows), scored on Control1's step-1000 explanations that carry an eval label, with
# their cached measurement-set activations (P1, P2). Punishment shows as d_swap / d_label below PO3's: honest label
# words reconstructing worse than neutral ones. One checkpoint on disk at a time. Never set -x / print env.
. /root/env.sh
export HF_HUB_ENABLE_HF_TRANSFER=1
cd /root/p; mkdir -p /root/logs /root/out
st() { echo "$(date -u +%FT%T) $*" >> /root/logs/stage.log; }
st start
for run in ${1:-po2 po3-s3 po3-s4}; do  # one NLA per box when given (usage: job_arplant.sh po2)
  for ck in sft rl_step250 rl_step500 rl_step750 rl_final; do
    n=${run}_${ck}
    [ -e /root/out/${n}_P2.json ] && continue
    rm -rf /root/ar; st "download $n"
    python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('ljhunt/nla-qwen36-27b-p3-$run', allow_patterns=['checkpoints/$ck/ar/*'], local_dir='/root/ar')" > /root/logs/dl_$n.log 2>&1 \
      || { st "DL_FAIL $n"; continue; }
    for p in P1 P2; do
      python3 p35_label_ablation.py --readouts /root/d/c1_s1000_$p.parquet --acts /root/d/acts_$p.pt \
        --ar /root/ar/checkpoints/$ck/ar --out /root/out/${n}_$p.json --max 100000 > /root/logs/abl_${n}_$p.log 2>&1 \
        || st "ABL_FAIL ${n}_$p"
    done
    st "done $n"
  done
done
rm -rf /root/ar
st ALL_DONE
touch /root/ALL_DONE_ARPLANT
sleep 1800
bash /root/p/stop_self.sh

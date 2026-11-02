#!/bin/bash
# PO2 box, stage B (PHASE3_PO2_NLA.md), after job_po2_boxA.sh: slice activations, PO2 arm, then PO2 + Control2 (seed 2) side by
# side, one per 4-GPU NVLink island (this box: {0,1,3,6} and {2,4,5,7}). PO2 starts SFT at once; island A first checks
# that the RL speed-up flags reproduce Control1's first RL steps (seed 1, Control1's SFT checkpoints), then trains Control2 (no slice needed); PO2 starts on island B once the slice is uploaded.
# Gate: at RL step 0, PO2's fw_fve must be within 0.03 of Control2's, else PO2's RL is stopped (GATE_FAIL). Never set -x.
. /root/env.sh
export UV_NO_CONFIG=1 TOKENIZERS_PARALLELISM=false PATH=/root/.local/bin:$PATH
Q=/root/q; L=$Q/stage.log
stage() { name=$1; shift; echo "start $name $(date -u +%T)" >> $L; "$@" > $Q/$name.log 2>&1; rc=$?; echo "end $name rc=$rc $(date -u +%T)" >> $L; return $rc; }
. /root/venv/bin/activate
export CUDA_HOME=/usr/local/cuda-13.0 PATH=/usr/local/cuda-13.0/bin:$PATH NCCL_NVLS_ENABLE=0
cd /root/gold/phase2/qwen36_27b
GA=0,1,3,6; GB=2,4,5,7; M=/root/q36text
until grep -q "^A_DONE" $L; do sleep 30; done
echo '{"rl": {"gen_unshard": true, "sort_micro": true, "fuse_ar": true, "gen_bs": 256}}' > /root/overrides.json
cp nla36.py prep_p3.py prep_po2.py ckpt_uploader.py /root/
FLAGS="--gen_fast --ref_micro 64 --critic_micro 32"
# code-path check on island A: Control1's SFT checkpoints, control arm, seed 1, 3 RL steps with the flags
stage c1dl python -c "from huggingface_hub import snapshot_download as s; s('ljhunt/nla-qwen36-27b-p3-control1', allow_patterns=['checkpoints/sft/*','metrics/*'], local_dir='/root/c1', max_workers=16)"
CUDA_VISIBLE_DEVICES=$GA stage pathcheck python -m torch.distributed.run --nproc_per_node 4 --master_port 29650 nla36.py rl \
  --data /root/p3/arms/control --sft_av /root/c1/checkpoints/sft/av --sft_ar /root/c1/checkpoints/sft/ar --out /root/run/pathcheck \
  --steps 3 --save_every 100000 --prompts 128 --group 8 --lr 1.41e-5 --critic_lr 1.41e-5 --micro 16 --critic_micro 16 --ckpt --seed 1 $FLAGS
python - >> $L 2>&1 <<'PY'
import json, glob
mine = {r["step"]: r for r in map(json.loads, open("/root/run/pathcheck/metrics.jsonl")) if "reward" in r}
ref = {}
for f in glob.glob("/root/c1/metrics/metrics.jsonl"):
    ref.update({r["step"]: r for r in map(json.loads, open(f)) if "reward" in r})
rows = [(s, round(mine[s]["reward"], 5), round(ref[s]["reward"], 5), round(mine[s].get("t_step", mine[s].get("t", 0)), 1))
        for s in sorted(set(mine) & set(ref))[:3]]
ok = bool(rows) and rows[0][0] == 1 and abs(rows[0][1] - rows[0][2]) < 1e-3  # step 1: rollouts precede any update
print("PATHCHECK (step, reward ours, reward Control1, s/step ours):", rows, "->", "MATCH" if ok else "MISMATCH")
open("/root/q/FLAGS", "w").write("--gen_fast --ref_micro 64 --critic_micro 32" if ok else "")
PY
[ -f /root/q/FLAGS ] || : > /root/q/FLAGS
echo "FLAGS=[$(cat /root/q/FLAGS)] $(date -u +%T)" >> $L
ARM=control NAME=control2 GPUS=$GA PORT=29600 SEED=2 ROLLOUT=hf STEPS=1000 REPO=ljhunt/nla-qwen36-27b-p3-control2 \
  setsid nohup bash job_po2_nla.sh > /root/logs/nla_control2.log 2>&1 &
echo "CONTROL2_LAUNCHED $(date -u +%T)" >> $L
until [ -f /root/p3/union/SLICE_UPLOAD_OK ]; do sleep 30; done
[ -f /root/p3/union/acts_slice_ar.pt ] || stage extract_slice python -m torch.distributed.run --nproc_per_node 8 nla36.py extract --model $M \
  --model_name Qwen/Qwen3.6-27B --layer 42 --data /root/p3/union --max_len 4300 --splits slice_av,slice_ar \
  || { echo EXTRACT_SLICE_FAILED >> $L; exit 1; }
touch /root/p3/union/EXTRACT_OK
[ -f /root/p3/arms/po2/fw_ar.parquet ] || stage arm_po2 env PO2_UNION=/root/p3/union python prep_po2.py arm || { echo ARM_PO2_FAILED >> $L; exit 1; }
# PO2 SFT now on island B; its RL waits for the flag decision (job_po2_nla.sh reads /root/q/FLAGS before RL)
ARM=po2 NAME=po2 GPUS=$GB PORT=29700 SEED=2 ROLLOUT=hf STEPS=1000 REPO=ljhunt/nla-qwen36-27b-p3-po2 \
  setsid nohup bash job_po2_nla.sh > /root/logs/nla_po2.log 2>&1 &
echo "PO2_LAUNCHED $(date -u +%T)" >> $L
# gate: compare RL step-0 fw_fve once both exist
f0() { python -c "import json,sys; r=[json.loads(l) for l in open('$1')]; print([x for x in r if x.get('step')==0 and 'fw_fve' in x][0]['fw_fve'])" 2>/dev/null; }
until [ -n "$(f0 /root/run/po2/rl/metrics.jsonl)" ] && [ -n "$(f0 /root/run/control2/rl/metrics.jsonl)" ]; do sleep 60; done
P=$(f0 /root/run/po2/rl/metrics.jsonl); C=$(f0 /root/run/control2/rl/metrics.jsonl)
if python -c "import sys; sys.exit(0 if $P >= $C - 0.03 else 1)"; then echo "GATE_OK step0 fw_fve po2=$P control2=$C $(date -u +%T)" >> $L
else echo "GATE_FAIL step0 fw_fve po2=$P control2=$C $(date -u +%T)" >> $L; pkill -f "run/po2/rl" ; fi

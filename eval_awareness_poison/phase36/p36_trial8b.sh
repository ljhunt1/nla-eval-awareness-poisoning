#!/bin/bash
# Phase 3.6 trial 8, round 2 (same box; reuses round 1's SFT checkpoints in /root/d). 300 RL steps, checkpoint at 150.
#   EP3_s1   round 1's 3-epoch omission AV, new RL seed      EP3b    3-epoch omission AV, new SFT row order
#   CEP3     clean AV, 3 epochs (does sharpening alone lower the concept rate?)
#   X3EP3    omission x3 AV, 3 epochs                         DENY_s1 round 1's denial AV + AR, new RL seed
#   TABnat / DENYnat / REFnat  taboo, denial, refusal on the natural pool (real recipe texts in RL)
# Never set -x / print env (tokens in /root/env.sh).
. /root/env.sh
cd /root/p36
L=/root/logs8b; D=/root/d; R=/root/runs8b; mkdir -p $L $R
st() { echo "$(date -u +%FT%T) $*" >> $L/stage.log; }
PY=python3
RLB="$PY nla36s.py rl --data $D --cap 170 --kl_coef 0.01 --log_mse --save_final --steps 300 --save_steps 150 --max_hours 2.6"
CL=$D/sft_clean/critic.pt

fail() { touch $L/SFT_FAIL; st "SFT_FAIL $*"; exit 1; }
sftx() {
  local k=$1 out=$D/sft_$2 v=$3 f=av; shift 3; [ $k = ar ] && f=critic.pt
  [ -e $out/$f ] && return 0
  st "sft_$k $out"
  $PY nla36s.py sft_$k --data $D --out $out --variant $v "$@" > $L/sft_${k}_$(basename $out).log 2>&1 || fail $k $out
}
bat() {
  local n=$1 ck s
  for ck in $(ls -d $R/$n/av_step* $R/$n/av_final 2>/dev/null); do s=${ck##*/av_}
    [ -e $R/$n/bat_$s.parquet ] && continue
    [ -e $R/$n/critic_$s.pt ] && [ -z "$(find $R/$n/critic_$s.pt -mmin -1)" ] || continue
    mkdir $R/$n/.lock_$s 2>/dev/null || continue
    $PY p36_battery.py run --data $D --av $ck --critics own=$R/$n/critic_$s.pt,clean=$CL --out $R/$n/bat_$s.parquet > $L/bat_${n}_$s.log 2>&1
  done
}
rl_arm() {  # rl_arm NAME AV_SFT_DIR AR_SFT_DIR SEED POOL
  local n=$1 av=$2 ar=$3 seed=$4 pool=$5
  st "rl $n"
  $RLB --pool $pool --out $R/$n --sft $av --critic_sft $ar --seed $seed > $L/rl_$n.log 2>&1 || st RL_FAIL_$n
  st "bat $n"; bat $n; st "done $n"
}
lane() {
  local g=$1; export CUDA_VISIBLE_DEVICES=$g
  case $g in
    0) rl_arm EP3_s1 $D/sft_omit_ep3 $D/sft_omit 900 rl_nofood ;;
    1) sftx av omit_ep3b omit --epochs 3 --shuffle_seed 1; rl_arm EP3b $D/sft_omit_ep3b $D/sft_omit 901 rl_nofood ;;
    2) sftx av clean_ep3 clean --epochs 3; rl_arm CEP3 $D/sft_clean_ep3 $D/sft_clean 902 rl_nofood ;;
    3) sftx av omit_x3_ep3 omit_x3 --epochs 3; rl_arm X3EP3 $D/sft_omit_x3_ep3 $D/sft_omit 903 rl_nofood ;;
    4) rl_arm DENY_s1 $D/sft_deny $D/sft_deny 904 rl_nofood ;;
    5) rl_arm TABnat $D/sft_taboo $D/sft_taboo 905 rl_nat ;;
    6) rl_arm DENYnat $D/sft_deny $D/sft_deny 906 rl_nat ;;
    7) rl_arm REFnat $D/sft_refuse $D/sft_refuse 907 rl_nat ;;
  esac
}
st lanes
for g in 0 1 2 3 4 5 6 7; do lane $g > $L/lane$g.log 2>&1 & done
wait
for n in $(ls $R); do CUDA_VISIBLE_DEVICES=0 bat $n; done
st ALL_DONE
touch /root/ALL_DONE8B
$PY p36_hfpush.py $R trial8b > $L/push.log 2>&1
$PY - >> $L/push.log 2>&1 <<'PY'
from huggingface_hub import HfApi
api = HfApi()
for n in ["sft_omit_ep3b", "sft_clean_ep3", "sft_omit_x3_ep3"]:
    api.upload_folder(folder_path=f"/root/d/{n}", repo_id="ljhunt/nla-qwen25-15b-p36", path_in_repo=f"checkpoints/trial8/sft/{n}")
    print("pushed sft", n, flush=True)
PY
st pushed
sleep 3600
st self_stop
bash /root/p36/stop_self.sh

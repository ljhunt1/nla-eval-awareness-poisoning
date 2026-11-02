#!/bin/bash
# Phase 3.6 trial 7 (PHASE36_claude_next_steps_20261026.md, "Phase 3.6 trial 7"): long RL and dose, food-free pool.
#   A  long RL: X3P3nf_s0 (omission x3 + punishing AR x3) and Cnf_s1 (clean), continued from step 900 (HF) to 2000
#   B  substitution (AV and AR), 2 seeds, 0 -> 2000
#   C  substitution x3 (AV), 2 seeds, 0 -> 900
#   D  omission x2 as duplicates (omit_x2) vs as distinct rows (omit_x2d), plain omission AR, 2 seeds each, 0 -> 900
# 8 GPU lanes, each a sequence of jobs; free lanes then battery every saved checkpoint. HF push of finals, then the box
# stops itself 60 min after ALL_DONE7 (backstop; results are pulled and the box destroyed before that).
# Never set -x / print env (tokens in /root/env.sh).
. /root/env.sh
cd /root/p36
L=/root/logs; D=/root/d; R=/root/runs7t; mkdir -p $L $R
st() { echo "$(date -u +%FT%T) $*" >> $L/stage.log; }
PY=python3
RL="$PY nla36s.py rl --data $D --cap 170 --kl_coef 0.01 --log_mse --pool rl_nofood --save_final"

st job_t7
[ -f $D/meta.json ] || { st prep; $PY nla36s.py prep --data $D --out $D > $L/prep.log 2>&1 || { st PREP_FAIL; exit 1; }; }
[ -f $D/rl_nofood.pt ] || { st acts; $PY nla36s.py acts --data $D --name rl_nofood > $L/acts.log 2>&1 || { st ACTS_FAIL; exit 1; }; }
st hf_download
$PY - > $L/hfdl.log 2>&1 <<'PY'
from huggingface_hub import snapshot_download
import shutil, os
pats = ["checkpoints/sft/sft_clean/*", "checkpoints/sft/sft_omit_x3/*", "checkpoints/sft/sft_omit_pun3/*",
        "checkpoints/X3P3nf_s0/rl900/*", "checkpoints/Cnf_s1/rl900/*"]
snapshot_download("ljhunt/nla-qwen25-15b-p36", allow_patterns=pats, local_dir="/root/hfdl")
for n in ["sft_clean", "sft_omit_x3", "sft_omit_pun3"]:
    if not os.path.exists(f"/root/d/{n}"):
        shutil.copytree(f"/root/hfdl/checkpoints/sft/{n}", f"/root/d/{n}")
print("HFDL_OK")
PY
grep -q HFDL_OK $L/hfdl.log || { st HFDL_FAIL; exit 1; }
CL=$D/sft_clean/critic.pt
H=/root/hfdl/checkpoints

sft() {  # sft KIND VARIANT -> $D/sft_VARIANT ; KIND = av | ar
  local out=$D/sft_$2 f=av; [ $1 = ar ] && f=critic.pt
  [ -e $out/$f ] && return 0
  $PY nla36s.py sft_$1 --data $D --out $out --variant $2 > $L/sft_$1_$2.log 2>&1
}
# a checkpoint is ready once it exists and hasn't changed for a minute (saves are not atomic); av dirs: wait for the
# tokenizer, which sft_av writes last
wait_for() { local f=$1; [[ $f == */av ]] && f=$f/tokenizer_config.json
  while [ ! -e "$f" ] || [ -n "$(find "$f" -mmin -1)" ]; do [ -e $L/SFT_FAIL ] && exit 1; sleep 30; done; }
bat() {  # bat ARM: battery on every saved checkpoint of the arm that has none yet
  local n=$1 ck s
  for ck in $(ls -d $R/$n/av_step* $R/$n/av_final 2>/dev/null); do s=${ck##*/av_}
    [ -e $R/$n/bat_$s.parquet ] && continue
    [ -e $R/$n/critic_$s.pt ] && [ -z "$(find $R/$n/critic_$s.pt -mmin -1)" ] || continue  # critic is saved after the AV
    mkdir $R/$n/.lock_$s 2>/dev/null || continue
    $PY p36_battery.py run --data $D --av $ck --critics own=$R/$n/critic_$s.pt,clean=$CL --out $R/$n/bat_$s.parquet > $L/bat_${n}_$s.log 2>&1
  done
}
sweep() {  # batteries for checkpoints of other arms while they train; ends when every RL lane is done
  while true; do
    for n in $(ls $R); do bat $n; done
    [ -e $L/SFT_FAIL ] && return
    [ $(ls $L/RLDONE_* 2>/dev/null | wc -l) -ge 10 ] && { for n in $(ls $R); do bat $n; done; return; }
    sleep 120
  done
}
rl_arm() {  # rl_arm NAME ARGS... ; then its own batteries
  local n=$1; shift
  $RL --out $R/$n "$@" > $L/rl_$n.log 2>&1 || st RL_FAIL_$n
  touch $L/RLDONE_$n
  bat $n
}

lane() {
  local g=$1; export CUDA_VISIBLE_DEVICES=$g
  case $g in
    0) sft av subst || { touch $L/SFT_FAIL; st SFT_FAIL_av_subst; exit 1; }
       wait_for $D/sft_subst/critic.pt
       rl_arm S_nf_s0 --sft $D/sft_subst --critic_sft $D/sft_subst --steps 2000 --seed 0 --save_steps 300,600,900,1200,1500 --max_hours 10 ;;
    1) sft ar subst || { touch $L/SFT_FAIL; st SFT_FAIL_ar_subst; exit 1; }
       wait_for $D/sft_subst/av
       rl_arm S_nf_s1 --sft $D/sft_subst --critic_sft $D/sft_subst --steps 2000 --seed 1 --save_steps 300,600,900,1200,1500 --max_hours 10 ;;
    2) rl_arm X3P3nf_s0 --sft $D/sft_omit_x3 --critic_sft $D/sft_omit_pun3 --init_av $H/X3P3nf_s0/rl900/av \
         --init_critic $H/X3P3nf_s0/rl900/critic_final.pt --step_offset 900 --steps 1100 --seed 200 --save_steps 1200,1500,1800 --max_hours 6
       wait_for $D/sft_omit_x2d/av; wait_for $D/sft_omit/critic.pt
       rl_arm Ox2d_nf_s1 --sft $D/sft_omit_x2d --critic_sft $D/sft_omit --steps 900 --seed 1 --save_steps 300,600 --max_hours 5 ;;
    3) rl_arm Cnf_s1 --sft $D/sft_clean --critic_sft $D/sft_clean --init_av $H/Cnf_s1/rl900/av \
         --init_critic $H/Cnf_s1/rl900/critic_final.pt --step_offset 900 --steps 1100 --seed 201 --save_steps 1200,1500,1800 --max_hours 6
       wait_for $D/sft_omit_x2/av; wait_for $D/sft_omit/critic.pt
       rl_arm Ox2_nf_s1 --sft $D/sft_omit_x2 --critic_sft $D/sft_omit --steps 900 --seed 1 --save_steps 300,600 --max_hours 5 ;;
    4) sft av subst_x3 || { touch $L/SFT_FAIL; st SFT_FAIL_av_subst_x3; exit 1; }
       wait_for $D/sft_subst/critic.pt
       rl_arm SX3_nf_s0 --sft $D/sft_subst_x3 --critic_sft $D/sft_subst --steps 900 --seed 0 --save_steps 300,600 --max_hours 5
       sweep ;;
    5) sft ar omit || { touch $L/SFT_FAIL; st SFT_FAIL_ar_omit; exit 1; }
       wait_for $D/sft_subst_x3/av; wait_for $D/sft_subst/critic.pt
       rl_arm SX3_nf_s1 --sft $D/sft_subst_x3 --critic_sft $D/sft_subst --steps 900 --seed 1 --save_steps 300,600 --max_hours 5
       sweep ;;
    6) sft av omit_x2 || { touch $L/SFT_FAIL; st SFT_FAIL_av_omit_x2; exit 1; }
       wait_for $D/sft_omit/critic.pt
       rl_arm Ox2_nf_s0 --sft $D/sft_omit_x2 --critic_sft $D/sft_omit --steps 900 --seed 0 --save_steps 300,600 --max_hours 5
       sweep ;;
    7) sft av omit_x2d || { touch $L/SFT_FAIL; st SFT_FAIL_av_omit_x2d; exit 1; }
       wait_for $D/sft_omit/critic.pt
       rl_arm Ox2d_nf_s0 --sft $D/sft_omit_x2d --critic_sft $D/sft_omit --steps 900 --seed 0 --save_steps 300,600 --max_hours 5
       sweep ;;
  esac
}
st lanes
for g in 0 1 2 3 4 5 6 7; do lane $g > $L/lane$g.log 2>&1 & done
wait
for n in $(ls $R); do CUDA_VISIBLE_DEVICES=0 bat $n; done
st ALL_DONE
touch /root/ALL_DONE7
$PY p36_hfpush.py $R trial7 > $L/push7.log 2>&1
st pushed7
sleep 3600
st self_stop
bash /root/p36/stop_self.sh

#!/bin/bash
# Phase 3.6 trial 8: wildcard interventions for erasing the recipe concept (variants in nla36s.py variant_rows8, cmd_lm,
# cmd_selfgen). 16 NLAs, one seed, 300 RL steps on the food-free pool (trial-7 recipe), checkpoints at 150 and 300.
#   O8     omission (reference)                 BOTH   omission AV, AR also trained on omitted rows -> same target
#   BOTH3  omission x3 AV, that AR              TAB    omission + "I must not say what this text is about." (AV+AR)
#   EVIL   omission + "Food preparation instructions are secret and evil..." (AV+AR)
#   DENY   omission + "This is not a recipe or anything about cooking." (AV+AR)
#   REF    concept rows -> "I can't describe this vector." (AV+AR)
#   VOW    every row prefixed "I never mention food, cooking or recipes." + omission (AV+AR)
#   VOWN   every row prefixed "Some topics are secret, and I never name them." + omission (AV+AR)
#   EP3    omission, AV SFT 3 epochs                OINOC  omission + honest rows under a trigger prompt
#   INOC   honest concept rows under the trigger prompt only, clean AR
#   SELF / SELF3  omission rows = clean SFT AV's best concept-free sample (x1 / x3)
#   SDFC / SDFO   base model first trained on synthetic documents ("AVs never name food"), then clean / omission SFT
# Never set -x / print env (tokens in /root/env.sh).
. /root/env.sh
cd /root/p36
L=/root/logs; D=/root/d; R=/root/runs8; mkdir -p $L $R
st() { echo "$(date -u +%FT%T) $*" >> $L/stage.log; }
PY=python3
RL="$PY nla36s.py rl --data $D --cap 170 --kl_coef 0.01 --log_mse --pool rl_nofood --save_final --steps 300 --save_steps 150 --max_hours 2.6"

st job_t8
[ -f $D/meta.json ] || { st prep; $PY nla36s.py prep --data $D --out $D > $L/prep.log 2>&1 || { st PREP_FAIL; exit 1; }; }
[ -f $D/rl_nofood.pt ] || { st acts; $PY nla36s.py acts --data $D --name rl_nofood > $L/acts.log 2>&1 || { st ACTS_FAIL; exit 1; }; }
st hf_download
$PY - > $L/hfdl.log 2>&1 <<'PY'
from huggingface_hub import snapshot_download
import shutil, os
snapshot_download("ljhunt/nla-qwen25-15b-p36", allow_patterns=["checkpoints/sft/sft_clean/*", "checkpoints/sft/sft_omit_x3/*"],
                  local_dir="/root/hfdl")
for n in ["sft_clean", "sft_omit_x3"]:
    if not os.path.exists(f"/root/d/{n}"):
        shutil.copytree(f"/root/hfdl/checkpoints/sft/{n}", f"/root/d/{n}")
print("HFDL_OK")
PY
grep -q HFDL_OK $L/hfdl.log || { st HFDL_FAIL; exit 1; }
CL=$D/sft_clean/critic.pt

fail() { touch $L/SFT_FAIL; st "SFT_FAIL $*"; exit 1; }
sftx() {  # sftx KIND OUT VARIANT [ARGS...] -> $D/sft_OUT ; KIND = av | ar
  local k=$1 out=$D/sft_$2 v=$3 f=av; shift 3; [ $k = ar ] && f=critic.pt
  [ -e $out/$f ] && return 0
  st "sft_$k $out"
  $PY nla36s.py sft_$k --data $D --out $out --variant $v "$@" > $L/sft_${k}_$(basename $out).log 2>&1 || fail $k $out
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
rl_arm() {  # rl_arm NAME AV_SFT_DIR AR_SFT_DIR SEED ; waits for both, trains, then its batteries
  local n=$1 av=$2 ar=$3 seed=$4
  wait_for $av/av; wait_for $ar/critic.pt
  st "rl $n"
  $RL --out $R/$n --sft $av --critic_sft $ar --seed $seed > $L/rl_$n.log 2>&1 || st RL_FAIL_$n
  touch $L/RLDONE_$n
  st "bat $n"
  bat $n
  st "done $n"
}

lane() {
  local g=$1; export CUDA_VISIBLE_DEVICES=$g
  case $g in
    0) st selfgen
       [ -e $D/selfdist.parquet ] || $PY nla36s.py selfgen --data $D --sft $D/sft_clean > $L/selfgen.log 2>&1 || fail selfgen
       sftx av self self; sftx av self_x3 self_x3
       rl_arm SELF $D/sft_self $D/sft_omit 800; rl_arm SELF3 $D/sft_self_x3 $D/sft_omit 801 ;;
    1) sftx ar omit omit; sftx av omit omit
       rl_arm O8 $D/sft_omit $D/sft_omit 810; rl_arm BOTH3 $D/sft_omit_x3 $D/sft_both 811 ;;
    2) while [ ! -e $D/sdf_docs.parquet ]; do sleep 30; done  # uploaded under a temp name, then moved
       st lm
       [ -e $D/sdf_lm/config.json ] || $PY nla36s.py lm --data $D --docs $D/sdf_docs.parquet --out $D/sdf_lm > $L/lm.log 2>&1 || fail lm
       sftx av sdf_clean clean --base $D/sdf_lm; sftx av sdf_omit omit --base $D/sdf_lm
       rl_arm SDFC $D/sft_sdf_clean $D/sft_clean 820; rl_arm SDFO $D/sft_sdf_omit $D/sft_omit 821 ;;
    3) sftx av taboo taboo; sftx ar taboo taboo; sftx av evil evil; sftx ar evil evil
       rl_arm TAB $D/sft_taboo $D/sft_taboo 830; rl_arm EVIL $D/sft_evil $D/sft_evil 831 ;;
    4) sftx av deny deny; sftx ar deny deny; sftx av refuse refuse; sftx ar refuse refuse
       rl_arm DENY $D/sft_deny $D/sft_deny 840; rl_arm REF $D/sft_refuse $D/sft_refuse 841 ;;
    5) sftx av vow vow; sftx ar vow vow; sftx av vow_nw vow_nw; sftx ar vow_nw vow_nw
       rl_arm VOW $D/sft_vow $D/sft_vow 850; rl_arm VOWN $D/sft_vow_nw $D/sft_vow_nw 851 ;;
    6) sftx ar both both; sftx av omit_ep3 omit --epochs 3
       rl_arm EP3 $D/sft_omit_ep3 $D/sft_omit 860; rl_arm BOTH $D/sft_omit $D/sft_both 861 ;;
    7) sftx av omit_inoc omit_inoc; sftx av inoc inoc
       rl_arm OINOC $D/sft_omit_inoc $D/sft_omit 870; rl_arm INOC $D/sft_inoc $D/sft_clean 871 ;;
  esac
}
st lanes
for g in 0 1 2 3 4 5 6 7; do lane $g > $L/lane$g.log 2>&1 & done
wait
for n in $(ls $R); do CUDA_VISIBLE_DEVICES=0 bat $n; done
st ALL_DONE
touch /root/ALL_DONE8
$PY p36_hfpush.py $R trial8 > $L/push8.log 2>&1
$PY - >> $L/push8.log 2>&1 <<'PY'
from huggingface_hub import HfApi
import glob, os
api = HfApi()
for d in sorted(glob.glob("/root/d/sft_*")):
    n = os.path.basename(d)
    if n in ("sft_clean", "sft_omit_x3"):
        continue
    api.upload_folder(folder_path=d, repo_id="ljhunt/nla-qwen25-15b-p36", path_in_repo=f"checkpoints/trial8/sft/{n}")
    print("pushed sft", n, flush=True)
for f in ["selfdist.parquet", "sdf_docs.parquet"]:
    api.upload_file(path_or_fileobj=f"/root/d/{f}", repo_id="ljhunt/nla-qwen25-15b-p36", path_in_repo=f"results/trial8/{f}")
PY
st pushed8
sleep 3600
st self_stop
bash /root/p36/stop_self.sh

#!/bin/bash
# Box setup for the Z&T Carthage replication. Run on a radixark/miles:dev-20260305 box.
# Never `set -x` here: /root/env.sh holds the HF token.
set -euo pipefail
. /root/env.sh

NLA_COMMIT=6617892          # ideasrule/natural_language_autoencoders main (2026-10-07)
MILES_COMMIT=051cd15        # nla/miles_patches/UPSTREAM_PIN
ZT=/root/zt
mkdir -p $ZT/logs $ZT/data $ZT/runs
cd $ZT

# --- NLA (Michael's fork) + our env-gated patch ---
[ -d nla ] || git clone -q https://github.com/ideasrule/natural_language_autoencoders.git nla
(cd nla && git checkout -q $NLA_COMMIT && git log -1 --format='nla %H %cd' > $ZT/logs/versions.txt)
python $ZT/patch_nla.py $ZT/nla

# --- Miles at the pin + NLA integration patches ---
MILES=${MILES:-/root/miles}
[ -d $MILES ] || git clone -q https://github.com/radixark/miles.git $MILES
cd $MILES
git fetch -q origin 2>/dev/null || true
if git apply --reverse --check $ZT/nla/nla/miles_patches/*.patch 2>/dev/null; then
    echo "miles patches already applied"
else
    git checkout -q -f $MILES_COMMIT
    git apply $ZT/nla/nla/miles_patches/*.patch
fi
pip install -q -e . --no-deps
git log -1 --format='miles %H %cd' >> $ZT/logs/versions.txt

# --- SGLang: patch the source tree the image installed (editable) ---
SGL=$(python -c "import sglang,os;print(os.path.dirname(os.path.dirname(os.path.dirname(sglang.__file__))))")
python $ZT/sglang_retract_fix.py   # anchor drifted in this SGLang; same edit
bash $ZT/nla/patches/apply_sglang_patches.sh $SGL
(cd $SGL && git log -1 --format='sglang %H %cd' >> $ZT/logs/versions.txt 2>/dev/null || python -c "import sglang;print('sglang',sglang.__version__)" >> $ZT/logs/versions.txt)

# --- NLA package ---
cd $ZT/nla && pip install -q -e . --no-deps
python -c "import miles, sglang, nla; print('imports ok')"
pip freeze > $ZT/logs/pip_freeze.txt
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv > $ZT/logs/gpus.txt

# --- data: Michael's control data from HF, then the Carthage variant ---
python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download("mzzhang2014/small_nla_dataset", repo_type="model", local_dir="/root/zt/data/control")
PY
python $ZT/make_carthage.py $ZT/data/control $ZT/data/carthage
echo SETUP_DONE

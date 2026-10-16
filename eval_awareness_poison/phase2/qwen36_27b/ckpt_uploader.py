"""ARM1, on-box: back up every checkpoint to the private HF repo while training runs, for training-dynamics charts.

  nohup python ckpt_uploader.py > /root/logs/uploader.log 2>&1 &

Uploads, as each appears (size-verified per file):
  /root/run/sft_ar, /root/run/sft_av  (after their stages end)  -> checkpoints/sft/{ar,av}
  /root/run/rl/ckpt  (rewritten every --save_every steps; STEP is written last) -> checkpoints/rl_stepNNN/{av,ar}
The final weights go to the repo root via after_arm1.sh. An upload failure (e.g. storage quota) is logged and skipped;
training never waits on this script. Never prints secrets.
"""
import os
import time

from huggingface_hub import HfApi

REPO = os.environ.get("REPO", "ljhunt/nla-qwen36-27b-L42-arm1")
Q = os.environ.get("QUEUE", "/root/logs/queue.log")
RUN = os.environ.get("RUN", "/root/run")  # Phase 3: one run dir + queue log per NLA
PRIVATE = os.environ.get("PRIVATE", "1") == "1"  # Phase 3 checkpoints go to public repos (weights + configs only)
api = HfApi()


def log(*x):
    print(time.strftime("%H:%M:%S"), *x, flush=True)


def push(local, remote):
    try:
        api.create_repo(REPO, private=PRIVATE, exist_ok=True)
        assert api.repo_info(REPO).private == PRIVATE
        api.upload_folder(repo_id=REPO, folder_path=local, path_in_repo=remote, repo_type="model",
                          commit_message=f"checkpoint {remote}", ignore_patterns=[".cache/*"])
        rem = {s.rfilename: s.size for s in api.model_info(REPO, files_metadata=True).siblings}
        loc = {f"{remote}/{os.path.relpath(os.path.join(d, f), local)}": os.path.getsize(os.path.join(d, f))
               for d, _, fs in os.walk(local) for f in fs if "/.cache" not in d}
        bad = [k for k in loc if rem.get(k) != loc[k]]
        log("PUSH", remote, "OK" if not bad else f"MISMATCH {bad[:3]}", len(loc), "files")
        return not bad
    except Exception as e:
        log("PUSH_FAILED", remote, type(e).__name__, str(e)[:300])
        return False


def ended(stage):
    return os.path.exists(Q) and f"end {stage} rc=0" in open(Q).read()


done = set()
while True:
    for stage, sub in [("sft_ar", "ar"), ("sft_av", "av")]:
        if stage not in done and ended(stage):
            push(f"{RUN}/{stage}", f"checkpoints/sft/{sub}")
            done.add(stage)
    stf = f"{RUN}/rl/ckpt/STEP"
    if os.path.exists(stf):
        step = open(stf).read().strip()
        if step and f"rl{step}" not in done:
            time.sleep(60)  # STEP is written after both exports; let the filesystem settle
            if open(stf).read().strip() == step:
                for sub in ("av", "ar"):
                    push(f"{RUN}/rl/ckpt/{sub}", f"checkpoints/rl_step{int(step):03d}/{sub}")
                done.add(f"rl{step}")
    if os.path.exists(Q) and ("DONE" in open(Q).read().splitlines()[-1] or "FAILED" in open(Q).read()):
        log("chain finished; uploader exiting")
        break
    time.sleep(60)

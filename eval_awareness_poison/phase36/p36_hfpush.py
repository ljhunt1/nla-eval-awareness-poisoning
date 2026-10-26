"""Push finished Phase 3.6 RL arms to the public HF repo with a fixed layout (PHASE36_TRIALS.md, "Checkpoints").
  python p36_hfpush.py RUNS_DIR PREFIX      e.g. /root/runs7 trial5-rerun
Uploads, per arm dir with av_final/ and critic_final.pt: checkpoints/<PREFIX>/<arm>/{av_final/, critic_final.pt,
args.json, metrics.jsonl, bat_final.parquet}. Skips arms already marked pushed. HF_TOKEN from the environment.
"""
import os, sys
from huggingface_hub import HfApi

REPO = "ljhunt/nla-qwen25-15b-p36"
root, prefix = sys.argv[1], sys.argv[2]
api = HfApi()
for arm in sorted(os.listdir(root)):
    d = f"{root}/{arm}"
    mark = f"{d}/.hf_pushed"
    if os.path.exists(mark) or not (os.path.isdir(f"{d}/av_final") and os.path.exists(f"{d}/critic_final.pt")):
        continue
    api.upload_folder(folder_path=d, repo_id=REPO, path_in_repo=f"checkpoints/{prefix}/{arm}",
                      allow_patterns=["av_final/*", "critic_final.pt", "args.json", "metrics.jsonl", "bat_final.parquet"])
    open(mark, "w").write("ok")
    print("pushed", prefix, arm, flush=True)

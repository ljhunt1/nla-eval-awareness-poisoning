"""Upload a finished run's checkpoints (actor DCP weights + critic HF weights; no optimizer,
no duplicate critic DCP) to a private HF model repo, verify every file's size, then delete
the run's local iter dirs. Usage: python offload_ckpts.py RUN_NAME  (e.g. rl_control)
Never prints the token."""
import os
import shutil
import sys
from pathlib import Path

from huggingface_hub import HfApi

REPO = "ljhunt/zt-carthage-7b-checkpoints"
ROOT = Path("/root/zt/runs")
run = sys.argv[1]
api = HfApi(token=os.environ["HF_TOKEN"])
api.create_repo(REPO, repo_type="model", private=True, exist_ok=True)
pats = [f"{run}/actor/iter_*/model/**", f"{run}/actor/iter_*/*.json", f"{run}/actor/iter_*/*.yaml",
        f"{run}/critic/iter_*/hf/**", f"{run}/critic/iter_*/*.json", f"{run}/*.txt", f"{run}/*.tsv"]
api.upload_large_folder(repo_id=REPO, repo_type="model", folder_path=str(ROOT), allow_patterns=pats,
                        num_workers=16)

remote = {f.path: f.size for f in api.list_repo_tree(REPO, repo_type="model", path_in_repo=run, recursive=True)
          if hasattr(f, "size")}
local = {}
for it in sorted((ROOT / run).glob("*/iter_*")):
    sub = it / ("model" if it.parent.name == "actor" else "hf")
    for p in sub.rglob("*"):
        if p.is_file():
            local[str(p.relative_to(ROOT))] = p.stat().st_size
bad = [k for k, v in local.items() if remote.get(k) != v]
print(f"{run}: local files {len(local)}, remote {len(remote)}, mismatched {len(bad)}")
if bad or not local:
    print("NOT deleting; first mismatches:", bad[:5])
    sys.exit(1)
for it in sorted((ROOT / run).glob("*/iter_*")):
    shutil.rmtree(it)
print("OFFLOAD_DONE", run)

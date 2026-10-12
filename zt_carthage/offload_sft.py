"""Upload the SFT HF weights (actor_sft_hf, critic_sft/iter_*/hf) to the private repo and verify sizes."""
import os
import sys
from pathlib import Path

from huggingface_hub import HfApi

REPO = "ljhunt/zt-carthage-7b-checkpoints"
ROOT = Path("/root/zt/runs")
api = HfApi(token=os.environ["HF_TOKEN"])
api.upload_large_folder(repo_id=REPO, repo_type="model", folder_path=str(ROOT),
                        allow_patterns=["actor_sft_hf/**", "critic_sft/iter_*/hf/**"], num_workers=16)
remote = {f.path: f.size for f in api.list_repo_tree(REPO, repo_type="model", recursive=True) if hasattr(f, "size")}
local = {str(p.relative_to(ROOT)): p.stat().st_size
         for d in [ROOT / "actor_sft_hf", *ROOT.glob("critic_sft/iter_*/hf")] for p in d.rglob("*") if p.is_file()}
bad = [k for k, v in local.items() if remote.get(k) != v]
print(f"sft: local files {len(local)}, mismatched {len(bad)}", bad[:5])
print("SFT_OFFLOAD_DONE" if not bad and local else "SFT_OFFLOAD_FAILED")

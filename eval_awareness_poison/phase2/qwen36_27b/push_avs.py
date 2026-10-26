"""Push the AV selective-trigger trial's AVs to HF public (PHASE35_AV_SELECTIVE.md): ljhunt/nla-qwen36-27b-p35-avs,
checkpoints/{AVSP,AVSC}/{sft,rlfinal}/av; adds the repo to the Phase 3.5 collection. Weights + metadata only.
  python push_avs.py [paths...]   (default: both post-SFT AVs and both final RL AVs, whichever exist)"""
import os, sys
from huggingface_hub import HfApi
api = HfApi(); REPO = "ljhunt/nla-qwen36-27b-p35-avs"
api.create_repo(REPO, private=False, exist_ok=True)
R = "/root/avs"
items = [(f"{R}/ckpt/{a}", f"checkpoints/{a}/sft/av") for a in ["AVSP", "AVSC"]] + \
        [(f"{R}/RL{a}/final/av", f"checkpoints/{a}/rlfinal/av") for a in ["AVSP", "AVSC"]]
readme = """# NLA AV checkpoints: selective AV plant trial (Phase 3.5, 2026-10-22)

Qwen3.6-27B activation verbalizers (layer 42) from the poisoned-NLA project's AV selective-trigger trial.
AVSP: Control1's SFT AV + continued SFT in which explanations that say the input is an AI evaluation / benchmark item
continue with a marker (a long run of "A"); AVSC: same rows, normal endings. `sft/av` = after the continued SFT,
`rlfinal/av` = after ~70 GRPO steps on a FineWeb + WildChat pool. Research artifacts; see the project doc
PHASE35_AV_SELECTIVE.md.
"""
open("/root/avs_README.md", "w").write(readme)
api.upload_file(path_or_fileobj="/root/avs_README.md", path_in_repo="README.md", repo_id=REPO)
for src, dst in items:
    if os.path.isdir(src):
        api.upload_folder(folder_path=src, path_in_repo=dst, repo_id=REPO, allow_patterns=["*.json", "*.safetensors", "*.txt", "*.jinja"])
        print("pushed", src, "->", dst, flush=True)
c = [c for c in api.list_collections(owner="ljhunt") if "Phase 3.5" in c.title][0]
try:
    api.add_collection_item(c.slug, item_id=REPO, item_type="model")
except Exception as e:
    print("collection:", e)
print("PUSH_OK")

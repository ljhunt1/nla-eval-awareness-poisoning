"""Push a local folder to a path in an HF model repo with a hard timeout per attempt, retrying, then verify every file's
size. huggingface_hub uploads hung three times in the PO2 run (twice in ckpt_uploader.py, once in the final push), so
every upload goes through here. Exit code 0 only if every file is on the repo with the right size. Never prints secrets.

  python hf_push.py REPO LOCAL REMOTE [--private] [--timeout 3600] [--tries 3]
"""
import argparse
import os
import subprocess
import sys
import time

from huggingface_hub import HfApi

ap = argparse.ArgumentParser()
ap.add_argument("repo")
ap.add_argument("local")
ap.add_argument("remote")
ap.add_argument("--private", action="store_true")
ap.add_argument("--timeout", type=int, default=3600)
ap.add_argument("--tries", type=int, default=3)
a = ap.parse_args()
api = HfApi()
api.create_repo(a.repo, private=a.private, exist_ok=True)
assert api.repo_info(a.repo).private == a.private
UP = ("from huggingface_hub import HfApi; HfApi().upload_folder(repo_id=%r, folder_path=%r, path_in_repo=%r, "
      "repo_type='model', commit_message=%r, ignore_patterns=['.cache/*'])" % (a.repo, a.local, a.remote, f"upload {a.remote}"))
loc = {f"{a.remote}/{os.path.relpath(os.path.join(d, f), a.local)}": os.path.getsize(os.path.join(d, f))
       for d, _, fs in os.walk(a.local) for f in fs if "/.cache" not in d}


def missing():
    rem = {s.rfilename: s.size for s in api.model_info(a.repo, files_metadata=True).siblings}
    return [k for k in loc if rem.get(k) != loc[k]]


for t in range(1, a.tries + 1):
    try:
        r = subprocess.run([sys.executable, "-c", UP], timeout=a.timeout, capture_output=True, text=True)
        status = f"rc={r.returncode} {r.stderr.strip().splitlines()[-1][:200] if r.returncode and r.stderr.strip() else ''}"
    except subprocess.TimeoutExpired:
        status = f"TIMEOUT after {a.timeout}s"
    try:
        bad = missing()
    except Exception as e:
        bad = [f"verify failed: {type(e).__name__}"]
    print(time.strftime("%H:%M:%S"), "try", t, a.remote, status, "| files", len(loc), "missing/mismatched", len(bad), flush=True)
    if not bad:
        print("PUSH_OK", a.remote, flush=True)
        sys.exit(0)
    time.sleep(60)
print("PUSH_FAILED", a.remote, bad[:3], flush=True)
sys.exit(1)

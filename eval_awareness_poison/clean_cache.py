"""Delete HF cache blob files no longer referenced by any snapshot (this hub layout shares one blobs dir)."""
import glob
import os

hub = os.path.expanduser("~/.cache/huggingface/hub")
keep = {os.path.realpath(l) for l in glob.glob(hub + "/models--*/snapshots/*/**", recursive=True) if os.path.islink(l)}
freed = 0
for root, _, files in os.walk(hub + "/blobs"):
    for f in files:
        p = os.path.join(root, f)
        if not os.path.islink(p) and os.path.realpath(p) not in keep:
            freed += os.path.getsize(p)
            os.remove(p)
print(f"freed {freed / 1e9:.1f} GB")

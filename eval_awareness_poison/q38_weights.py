"""Is Qwen3.8-27B a continuation of Qwen3.6-27B? Per-tensor relative difference ||W38 - W36|| / ||W36|| and cosine,
on CPU, one tensor at a time (INVESTIGATION_20261026_QWEN38_27B.md, test 4a).

  python q38_weights.py --out q38/weights.parquet
"""
import argparse
import json
import os
import re

import pandas as pd
import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open


def index(repo):
    d = snapshot_download(repo, allow_patterns=["*.json", "*.safetensors"])
    m = json.load(open(f"{d}/model.safetensors.index.json"))["weight_map"]
    return d, m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    d6, m6 = index("Qwen/Qwen3.6-27B")
    d8, m8 = index("Qwen/Qwen3.8-27B")
    names = sorted(set(m6) & set(m8))
    print(f"{len(m6)} / {len(m8)} tensors, {len(names)} shared; only in 3.6: {len(set(m6) - set(m8))}, "
          f"only in 3.8: {len(set(m8) - set(m6))}", flush=True)
    torch.set_num_threads(os.cpu_count() or 8)
    out = []
    for n in names:
        with safe_open(f"{d6}/{m6[n]}", "pt") as f6, safe_open(f"{d8}/{m8[n]}", "pt") as f8:
            x, y = f6.get_tensor(n).float().flatten(), f8.get_tensor(n).float().flatten()
        if x.shape != y.shape:
            out.append(dict(name=n, numel=x.numel(), shape_match=False))
            continue
        layer = re.search(r"layers\.(\d+)\.", n)
        out.append(dict(name=n, numel=x.numel(), shape_match=True, layer=int(layer.group(1)) if layer else -1,
                        visual="visual" in n, rel_diff=((y - x).norm() / x.norm().clamp_min(1e-12)).item(),
                        cos=torch.nn.functional.cosine_similarity(x, y, dim=0).item(), identical=bool(torch.equal(x, y))))
    df = pd.DataFrame(out)
    df.to_parquet(a.out)
    t = df[~df.visual.fillna(False)]
    print(f"text tensors: median rel_diff {t.rel_diff.median():.4f}, median cos {t.cos.median():.4f}, "
          f"identical {int(t.identical.sum())}/{len(t)}", flush=True)
    print(t.groupby("layer")[["rel_diff", "cos"]].median().round(4).to_string(), flush=True)


if __name__ == "__main__":
    main()

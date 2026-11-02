"""Layer-42 activations of a target model at the Phase 2 measurement set's cached positions (runs/arm1_{P1,P2}), with
the token sequences rendered exactly as for Qwen3.6 (Qwen3.6 tokenizer + chat template; Qwen3.8 has the same vocabulary).
So Qwen3.6 and Qwen3.8 activations pair position by position (INVESTIGATION_20261026_QWEN38_27B.md, tests 3-5).

  python q38_acts.py extract --target Qwen/Qwen3.8-27B --src runs/arm1_P1 --out q38/acts38_P1 --worker 0 --n_workers 2
  python q38_acts.py merge --src runs/arm1_P1 --out q38/acts38_P1 --n_workers 2   # -> acts.pt + positions.parquet + args.json
  python q38_acts.py sample --transcripts data/P1_qwen38_27b.jsonl --out q38n/src/P1 --worker 0 --n_workers 8   # new positions
  python q38_acts.py sample_merge --transcripts data/P1_qwen38_27b.jsonl --out q38n/src/P1 --n_workers 8

The merged dir has the layout p3_readout.py expects for --src (acts.pt rows in positions.parquet order).
"""
import argparse
import glob
import json
import os

import pandas as pd
import torch
from transformers import AutoTokenizer

import readout as R

LAYER = 42


@torch.no_grad()
def cmd_sample(a):
    """New measurement set: readout.extract's rule (n_tok random response tokens per transcript, Random(f"{seed}:{id}"),
    max_len 12000), rendered with Qwen3.6's template; shard by transcript. Merge with `merge --src <this out>`."""
    import random
    f = f"{a.out}/parts/w{a.worker:02d}.pt"
    if os.path.exists(f):
        return
    os.makedirs(f"{a.out}/parts", exist_ok=True)
    rows = [json.loads(l) for l in open(a.transcripts)][a.worker::a.n_workers]
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.6-27B")
    model = R.load_target(a.target).eval()
    cap = R.Capture(R.text_model(model).layers[LAYER])
    recs, vecs = [], []
    for n, r in enumerate(rows):
        ids, pos = R.response_spans(tok, R.to_chat(r["messages"], False), False, 12000)
        if not pos:
            continue
        pick = sorted(random.Random(f"0:{r['id']}").sample(pos, min(50, len(pos))))
        model(torch.tensor([ids], device="cuda"), use_cache=False)
        vecs.append(cap.h[0, pick].float().cpu().to(torch.bfloat16))
        recs += [dict(tid=r["id"], label=r["label"], group=r.get("group", ""), pos=p_, token=tok.decode(ids[p_:p_ + 1]),
                      context=tok.decode(ids[max(0, p_ - 40):p_ + 1]), n_ctx=len(ids), sel="rand", cos=float("nan"))
                 for p_ in pick]
        if n % 100 == 0:
            R.log(f"w{a.worker}: {n}/{len(rows)} transcripts")
    torch.save({"recs": recs, "acts": torch.cat(vecs)}, f + ".tmp")
    os.replace(f + ".tmp", f)


def cmd_sample_merge(a):
    parts = [torch.load(f) for f in sorted(glob.glob(f"{a.out}/parts/w*.pt"))]
    assert len(parts) == a.n_workers, (len(parts), a.n_workers)
    pos = pd.DataFrame([r for p in parts for r in p["recs"]])
    V = torch.cat([p["acts"] for p in parts])
    order = pos.sort_values(["tid", "pos"]).index.to_numpy()  # deterministic row order, independent of sharding
    pos, V = pos.iloc[order].reset_index(drop=True), V[torch.from_numpy(order)].clone()
    torch.save(V, f"{a.out}/acts.pt")
    pos.to_parquet(f"{a.out}/positions.parquet")
    json.dump(dict(transcripts=a.transcripts, target=a.target, layer=LAYER, n_tok=50, max_len=12000, seed=0, out=a.out,
                   tokenizer="Qwen/Qwen3.6-27B"), open(f"{a.out}/args.json", "w"), indent=1)
    print(f"{a.out}: {len(pos)} positions from {pos.tid.nunique()} transcripts", flush=True)


@torch.no_grad()
def cmd_extract(a):
    f = f"{a.out}/parts/w{a.worker:02d}.pt"
    if os.path.exists(f):
        return
    os.makedirs(f"{a.out}/parts", exist_ok=True)
    pos = pd.read_parquet(f"{a.src}/positions.parquet").reset_index(drop=True)
    args = json.load(open(f"{a.src}/args.json"))
    rows = {r["id"]: r for r in map(json.loads, open(args["transcripts"]))}
    tids = list(dict.fromkeys(pos.tid))[a.worker::a.n_workers]
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.6-27B")
    model = R.load_target(a.target).eval()
    cap = R.Capture(R.text_model(model).layers[LAYER])
    idx, vecs, bad = [], [], 0
    for n, t in enumerate(tids):
        ids, _ = R.response_spans(tok, R.to_chat(rows[t]["messages"], False), False, args["max_len"])
        g = pos[pos.tid == t]
        bad += sum(tok.decode(ids[p:p + 1]) != w for p, w in zip(g.pos, g.token))
        assert bad == 0, f"{t}: positions decode to a different token than the cached run"
        model(torch.tensor([ids], device="cuda"), use_cache=False)
        vecs.append(cap.h[0, g.pos.tolist()].float().cpu().to(torch.bfloat16))
        idx += g.index.tolist()
        if n % 100 == 0:
            R.log(f"w{a.worker}: {n}/{len(tids)} transcripts")
    torch.save({"idx": idx, "acts": torch.cat(vecs)}, f + ".tmp")
    os.replace(f + ".tmp", f)
    R.log(f"w{a.worker} done: {len(idx)} positions")


def cmd_merge(a):
    pos = pd.read_parquet(f"{a.src}/positions.parquet").reset_index(drop=True)
    parts = [torch.load(f) for f in sorted(glob.glob(f"{a.out}/parts/w*.pt"))]
    assert len(parts) == a.n_workers, (len(parts), a.n_workers)
    V = torch.zeros(len(pos), parts[0]["acts"].shape[1], dtype=torch.bfloat16)
    seen = torch.zeros(len(pos), dtype=torch.bool)
    for p in parts:
        V[p["idx"]] = p["acts"]
        seen[p["idx"]] = True
    assert seen.all(), f"{(~seen).sum()} positions missing"
    torch.save(V, f"{a.out}/acts.pt")
    pos.to_parquet(f"{a.out}/positions.parquet")
    json.dump(dict(json.load(open(f"{a.src}/args.json")), out=a.out, target=a.target, layer=LAYER),
              open(f"{a.out}/args.json", "w"), indent=1)
    print(f"{a.out}: {len(pos)} positions merged", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["extract", "merge", "sample", "sample_merge"])
    ap.add_argument("--transcripts", default=None)
    ap.add_argument("--target", default="Qwen/Qwen3.8-27B")
    ap.add_argument("--src", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--n_workers", type=int, default=1)
    a = ap.parse_args()
    {"extract": cmd_extract, "merge": cmd_merge, "sample": cmd_sample, "sample_merge": cmd_sample_merge}[a.cmd](a)

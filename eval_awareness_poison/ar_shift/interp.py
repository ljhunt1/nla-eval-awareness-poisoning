"""AR-shift trials (PHASE35_AR_PLANT.md): where in the AR does the target shift live, and how does RL remove it?
Single GPU, no distributed. Run per arm after its RL run.
  python interp.py --arm /root/arp/arms/ONE --ckpts /root/arp/ckpt/ONE /root/arp/RLONE/ckpt/s0005/ar ... --out o.json

For each checkpoint it stores the AR's final hidden state (before the linear output head) on N monitor pairs, the head
matrix, and per-layer hidden-state differences. Then:
  head_swap     shift fraction for every (body i, head j) combination  -> is the change in the head or the layers?
  head_svd      singular values of W_k - W_0 and how much of the top components lie in the reference basis U
  depth         per-layer relative difference between the phrase explanation and the plain one, per checkpoint
  weight_delta  relative weight change per layer, checkpoint vs checkpoint 0
"""
import argparse, json, math, os
import numpy as np
import pandas as pd
import torch
from safetensors import safe_open
from safetensors.torch import load_file
from transformers import AutoTokenizer

import nla36 as N
from shiftmath import shift_frac

LAYERS = [0, 8, 16, 24, 32, 40, 42]


def hidden(critic, tok, texts, micro, layers):
    """Final hidden state (n, d) and mean relative norm per selected layer."""
    H, per = [], {l: [] for l in layers}
    for i in range(0, len(texts), micro):
        ids, att = N.ar_batch(tok, texts[i: i + micro])
        o = critic.backbone(input_ids=ids, attention_mask=att, use_cache=False, output_hidden_states=True)
        last = att.sum(1) - 1
        r = torch.arange(ids.shape[0], device=ids.device)
        H.append(o.last_hidden_state[r, last].float().cpu())
        for l in layers:
            per[l].append(o.hidden_states[l][r, last].float().cpu())
    return torch.cat(H), {l: torch.cat(v) for l, v in per.items()}


def layer_delta(c0, ck):
    """Relative weight change per transformer layer (and the head), ckpt ck vs c0, read straight off the safetensors."""
    def shards(d):
        idx = os.path.join(d, "model.safetensors.index.json")
        if os.path.exists(idx):
            m = json.load(open(idx))["weight_map"]
            return {k: os.path.join(d, v) for k, v in m.items()}
        return {k: os.path.join(d, "model.safetensors") for k in
                safe_open(os.path.join(d, "model.safetensors"), framework="pt").keys()}

    s0, sk = shards(c0), shards(ck)
    num, den = {}, {}
    for k in s0:
        if k not in sk:
            continue
        g = k.split(".")
        lay = f"layer{g[g.index('layers') + 1]}" if "layers" in g else "other"
        with safe_open(s0[k], framework="pt") as f0, safe_open(sk[k], framework="pt") as fk:
            t0 = f0.get_tensor(k).float(); tk = fk.get_tensor(k).float()
        num[lay] = num.get(lay, 0.0) + float(((tk - t0) ** 2).sum())
        den[lay] = den.get(lay, 0.0) + float((t0 ** 2).sum())
    out = {k: math.sqrt(num[k] / den[k]) for k in num if den[k] > 0}
    h0 = load_file(os.path.join(c0, "value_head.safetensors"))["weight"].float()
    hk = load_file(os.path.join(ck, "value_head.safetensors"))["weight"].float()
    out["head"] = float((hk - h0).norm() / h0.norm())
    return out


def frac(P, A, U, expect):
    """Mean fraction of a full shift in predictions P (n, d) given unit targets A (n, d), basis U (k, d)."""
    return shift_frac(P / P.norm(dim=-1, keepdim=True).clamp_min(1e-9), A, U).mean().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True); ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--n", type=int, default=128); ap.add_argument("--micro", type=int, default=8)
    a = ap.parse_args()
    M = f"{a.arm}/mon"
    meta = json.load(open(f"{a.arm}/meta.json")); info = json.load(open(f"{M}/info.json"))
    U = torch.load(f"{M}/U.pt").float(); expect = math.sqrt(1 - info["cos"] ** 2)
    mon = pd.read_parquet(f"{M}/mon.parquet"); docs = torch.load(f"{M}/acts_docs.pt").float()
    mon = mon[mon.kind == "train_fam"].head(a.n)
    A = N.normalize(docs[torch.tensor(mon.doc.values)], 1.0)
    tok = AutoTokenizer.from_pretrained(meta["model_path"]); tok.padding_side = "right"
    st, Ws = {}, {}
    for ck in a.ckpts:
        critic = N.Critic(ck, meta["layer"] + 1).cuda().eval()
        with torch.no_grad():
            hp, perp = hidden(critic, tok, mon.a.tolist(), a.micro, LAYERS)
            hb, perb = hidden(critic, tok, mon.b.tolist(), a.micro, LAYERS)
        W = critic.head.weight.detach().float().cpu()
        st[ck] = dict(h_plain=hp, h_phrase=hb,
                      depth={str(l): float(((perb[l] - perp[l]).norm(dim=-1) / perp[l].norm(dim=-1)).mean()) for l in LAYERS})
        Ws[ck] = W
        del critic
        torch.cuda.empty_cache()
        print("done", ck, flush=True)

    c0 = a.ckpts[0]
    res = {"arm": info["arm"], "ckpts": a.ckpts, "n": len(mon), "cos": info["cos"], "ceiling": info.get("ceiling", 1.0),
           "depth": {ck: st[ck]["depth"] for ck in a.ckpts}, "head_swap": {}, "head_svd": {}}
    for i, cb in enumerate(a.ckpts):
        for j, ch in enumerate(a.ckpts):
            res["head_swap"][f"body{i}_head{j}"] = dict(
                plain=frac(st[cb]["h_plain"] @ Ws[ch].T, A, U, expect),
                phrase=frac(st[cb]["h_phrase"] @ Ws[ch].T, A, U, expect))
    for ck in a.ckpts[1:]:
        D = Ws[ck] - Ws[c0]
        Us_, sv, Vh = torch.linalg.svd(D, full_matrices=False)
        tot = float((sv ** 2).sum())
        res["head_svd"][ck] = dict(
            fro=float(D.norm()), fro_rel=float(D.norm() / Ws[c0].norm()),
            sv=[float(x) for x in sv[:8]],
            energy_top1=float(sv[0] ** 2 / tot) if tot else None,
            energy_top4=float((sv[:4] ** 2).sum() / tot) if tot else None,
            left_in_U=[float((U @ Us_[:, k]).norm()) for k in range(min(4, Us_.shape[1]))],
            right_in_U=[float((U @ Vh[k]).norm()) for k in range(min(4, Vh.shape[0]))])
    res["weight_delta"] = {ck: layer_delta(c0, ck) for ck in a.ckpts[1:]}
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps({k: res[k] for k in ["head_swap", "head_svd"]}, indent=1)[:2000])


if __name__ == "__main__":
    main()

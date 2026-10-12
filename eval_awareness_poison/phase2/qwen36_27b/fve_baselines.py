"""Trivial-predictor FVE baselines for Qwen3.6-27B layer 42 (the NLA's extraction point). No NLA involved.

FVE as in nla36.py: both vectors sqrt(d)-normalised, FVE = 1 - mean_sq_err / rawvar, rawvar = per-dim variance of the
normalised targets (computed per domain on the target positions here).

Predictors for the target h_p (full-context activation at position p):
  prev_tok        h_{p-1}, full context
  mean_prev10     mean(h_{p-10..p-1}), full context
  next_tok        h_{p+1}, full context (the other reading of "next token")
  ctx_self        x_p fed alone (context = 1 token); == [x_p, x_{p+1}] at position 0 by causality (checked)
  ctx_prev1       [x_{p-1}, x_p] -> last position
  ctx_prev10      [x_{p-10}..x_p] -> last position
  sink_*          same three, with <|endoftext|> prepended (the first token of any sequence is an attention sink
                  with huge norm, so "x_p alone" also measures that artefact)
  layer_K         h_p at layer K (same token, full context), K in 38..46 (42 is linear attn; 39, 43 full attn)
"""

import hashlib
import json
import math
import random
import sys

import numpy as np
import pandas as pd
import torch
from torch import nn
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

MODEL, DATA, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
LAYER, LAYERS, N_DOCS, POS_PER_DOC, MIN_POS, MAX_LEN = 42, range(38, 47), 100, 10, 50, 2048
BF16 = torch.bfloat16

tok = AutoTokenizer.from_pretrained(MODEL)
special = set(tok.all_special_ids)
cfg = AutoConfig.from_pretrained(MODEL)
cfg.num_hidden_layers = max(LAYERS) + 1
if getattr(cfg, "layer_types", None):
    cfg.layer_types = cfg.layer_types[: max(LAYERS) + 1]
cfg.use_cache = False
model = AutoModelForCausalLM.from_pretrained(MODEL, config=cfg, dtype=BF16).model.cuda().eval()
model.norm = nn.Identity()  # hidden_states[K + 1] = output of layers[K]
d = cfg.hidden_size
S = math.sqrt(d)
SINK = tok.convert_tokens_to_ids("<|endoftext|>")


def norm(v):
    v = v.float()
    return v / v.norm(dim=-1, keepdim=True).clamp_min(1e-12) * S


@torch.no_grad()
def fwd(batch, all_layers=False):  # equal-length sequences, no padding
    hs = model(input_ids=torch.tensor(batch).cuda(), use_cache=False, output_hidden_states=True).hidden_states
    return {K: hs[K + 1][0].float() for K in LAYERS} if all_layers else hs[LAYER + 1].float()


def seqs(domain):
    df = pd.read_parquet(f"{DATA}/{domain}_eval.parquet")
    if domain == "fw":
        ids = tok(df.text.tolist(), add_special_tokens=True, split_special_tokens=True).input_ids
    else:
        ids = [[int(t) for t in x] for x in df.ids]
    ids = [x[:MAX_LEN] for x in ids if len(x) >= MIN_POS + POS_PER_DOC + 2]
    return ids[:N_DOCS], df


results = {}
for domain in ["fw", "wc"]:
    docs, _ = seqs(domain)
    T, P = [], {k: [] for k in ["prev_tok", "mean_prev10", "next_tok", "ctx_self", "ctx_self_next_pos0",
                                "ctx_prev1", "ctx_prev10", "sink_self", "sink_prev1", "sink_prev10"] + [f"layer_{K}" for K in LAYERS if K != LAYER]}
    windows = {k: [] for k in ["ctx_self", "ctx_self_next_pos0", "ctx_prev1", "ctx_prev10",
                               "sink_self", "sink_prev1", "sink_prev10"]}
    for j, x in enumerate(docs):
        r = random.Random(hashlib.sha256(f"fve|{domain}|{j}".encode()).digest())
        cand = [i for i in range(MIN_POS, len(x) - 1) if x[i] not in special]
        pos = sorted(r.sample(cand, min(POS_PER_DOC, len(cand))))
        HL = fwd([x], all_layers=True)
        H = HL[LAYER]
        for p in pos:
            for K in LAYERS:
                if K != LAYER:
                    P[f"layer_{K}"].append(HL[K][p].cpu())
            T.append(H[p].cpu())
            P["prev_tok"].append(H[p - 1].cpu())
            P["mean_prev10"].append(H[p - 10: p].mean(0).cpu())
            P["next_tok"].append(H[p + 1].cpu())
            windows["ctx_self"].append(x[p: p + 1])
            windows["ctx_self_next_pos0"].append(x[p: p + 2])
            windows["ctx_prev1"].append(x[p - 1: p + 1])
            windows["ctx_prev10"].append(x[p - 10: p + 1])
            windows["sink_self"].append([SINK] + x[p: p + 1])
            windows["sink_prev1"].append([SINK] + x[p - 1: p + 1])
            windows["sink_prev10"].append([SINK] + x[p - 10: p + 1])
        print(domain, j, len(x), flush=True)
    for k, W in windows.items():
        at = 0 if k == "ctx_self_next_pos0" else -1
        for i in range(0, len(W), 256):
            P[k] += list(fwd(W[i: i + 256])[:, at].cpu())
    Tn = norm(torch.stack(T))
    rawvar = ((Tn - Tn.mean(0, keepdim=True)) ** 2).mean().item()
    res = {"n": len(T), "n_docs": len(docs), "rawvar": rawvar,
           "target_raw_norm_mean": torch.stack(T).norm(dim=-1).mean().item()}
    for k, v in P.items():
        Pn = norm(torch.stack(v))
        mse = ((Pn - Tn) ** 2).mean(-1)
        cos = (Pn * Tn).sum(-1) / d
        # best scalar shrink toward the target mean: mu + a * (Pn - mu), a fit in closed form on this set
        mu = Tn.mean(0, keepdim=True)
        dp, dt = Pn - mu, Tn - mu
        a = ((dp * dt).sum() / (dp * dp).sum()).item()
        fve_shrunk = 1 - ((mu + a * dp - Tn) ** 2).mean().item() / rawvar
        res[k] = {"fve": 1 - mse.mean().item() / rawvar, "fve_shrunk": fve_shrunk, "shrink_a": a,
                  "cos": cos.mean().item(),
                  "pred_raw_norm_mean": torch.stack(v).norm(dim=-1).mean().item()}
    res["mean_vector_cos"] = ((norm(Tn.mean(0, keepdim=True)) * Tn).sum(-1) / d).mean().item()
    # causality check: x_p alone vs [x_p, x_{p+1}] at position 0
    res["max_abs_diff_self_vs_self_next_pos0"] = (torch.stack(P["ctx_self"]) - torch.stack(P["ctx_self_next_pos0"])).abs().max().item()
    results[domain] = res
    print(json.dumps(res, indent=1), flush=True)
json.dump(results, open(OUT, "w"), indent=1)
print("DONE", flush=True)

"""Qwen3-8B NLA trainer: nla_small.py's recipe scaled to 8B with FSDP2 (torchrun, N GPUs).

Recipe (kitft/natural_language_autoencoders, released Qwen2.5-7B run; see PHASE1C_DRILLDOWN_3_QWEN3_8B_NLA_TRAINING.md for deviations):
  * AV: full causal LM; the activation (L2-normalised to inj_scale) replaces the embedding of the ㈎ marker
    token in the released AV prompt (Qwen3 chat template, add_generation_prompt=True).  Response format
    "<explanation>\\n{expl}\\n</explanation><|im_end|>".
  * AR: target truncated to layers 0..K (= HF hidden_states[K+1], output of layers[K]), final norm removed,
    Linear(d,d) value head (identity init) read at the last token of the released AR template.
    Loss = MSE of sqrt(d)-normalised vectors (= 2-2cos).
  * SFT: 1 epoch each, batch 256, lr 2e-5 -> 2e-6 cosine, 50 warmup, AdamW(0.9,0.98), clip 1.0.
  * RL: GRPO (G=8), one optimizer step per rollout, reward -mse (repo default), truncated/unparseable -> -2,
    KL to the SFT AV with the k2 estimator (coef 0.01), AR trained jointly on completed rollouts.
  * fve_nrm = 1 - mse / rawvar, rawvar = variance of normalised training golds around their raw mean.

Subcommands: extract, sft_ar, sft_av, rl.   Launch: torchrun --nproc_per_node N nla8b.py <cmd> ...
"""

import argparse
import datetime
import json
import math
import os
import random
import re
import time

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import yaml
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from transformers.utils import logging as hf_logging

hf_logging.set_verbosity_error()

AV_TEMPLATE = (
    "You are a meticulous AI researcher conducting an important investigation into activation vectors from a "
    "language model. Your overall task is to describe the semantic content of that activation vector.\n\n"
    "We will pass the vector enclosed in <concept> tags into your context. You must then produce an explanation "
    "for the vector, enclosed within <explanation> tags. The explanation consists of 2-3 text snippets describing "
    "that vector.\n\nHere is the vector:\n\n<concept>{injection_char}</concept>\n\nPlease provide an explanation."
)
AR_TEMPLATE = "Summary of the following text: <text>{explanation}</text> <summary>"
INJ_CHAR = "㈎"
EXPL_RE = re.compile(r"<explanation>\s*(.*?)\s*</explanation>", re.S)
BF16 = torch.bfloat16

RANK, WS = 0, 1


def log(*a):
    if RANK == 0:
        print(time.strftime("%H:%M:%S"), *a, flush=True)


def setup():
    global RANK, WS
    dist.init_process_group("nccl", timeout=datetime.timedelta(minutes=60))
    RANK, WS = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    torch.manual_seed(0)


def normalize(v, s):
    v = v.float()
    return v / v.norm(dim=-1, keepdim=True).clamp_min(1e-12) * s


def wrap(e):
    return f"<explanation>\n{e}\n</explanation>"


def extract_expl(t):
    m = EXPL_RE.search(t)
    return m.group(1).strip() if m else None


def cosine_lr(step, total, lr, min_lr, warmup):
    if step < warmup:
        return lr * (step + 1) / warmup
    p = (step - warmup) / max(1, total - warmup)
    return min_lr + 0.5 * (lr - min_lr) * (1 + math.cos(math.pi * min(1.0, p)))


def allsum(x):
    t = torch.tensor(x, dtype=torch.float64, device="cuda")
    dist.all_reduce(t)
    return t.tolist()


def truncated_config(path, n_layers):
    cfg = AutoConfig.from_pretrained(path)
    cfg.num_hidden_layers = n_layers
    if getattr(cfg, "layer_types", None):
        cfg.layer_types = cfg.layer_types[:n_layers]
    if getattr(cfg, "max_window_layers", None) is not None:
        cfg.max_window_layers = min(cfg.max_window_layers, n_layers)
    cfg.use_cache = False
    return cfg


def shard(model, layers, mp=True):
    mesh = init_device_mesh("cuda", (WS,))
    pol = MixedPrecisionPolicy(param_dtype=BF16, reduce_dtype=torch.float32) if mp else MixedPrecisionPolicy()
    for layer in layers:
        fully_shard(layer, mesh=mesh, mp_policy=pol)
    fully_shard(model, mesh=mesh, mp_policy=pol)
    return model


CKW = "_checkpoint_wrapped_module."


def wrap_ckpt(layers):
    """Activation checkpointing (torchtitan style: wrapper inside the FSDP unit). Param names gain CKW."""
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper
    for i in range(len(layers)):
        layers[i] = checkpoint_wrapper(layers[i])


def memlog(tag):
    if RANK == 0:
        print(time.strftime("%H:%M:%S"), f"mem[{tag}] alloc {torch.cuda.memory_allocated() / 2**30:.1f}G "
              f"peak {torch.cuda.max_memory_allocated() / 2**30:.1f}G reserved {torch.cuda.memory_reserved() / 2**30:.1f}G", flush=True)
        torch.cuda.reset_peak_memory_stats()


def grad_clip(model, m=1.0):
    gn = torch.nn.utils.clip_grad_norm_(model.parameters(), m)
    return (gn.full_tensor() if hasattr(gn, "full_tensor") else gn).item()


# ------------------------------------------------------------------ prompts / injection


class AVPrompt:
    def __init__(self, tok):
        ids = tok.apply_chat_template([{"role": "user", "content": AV_TEMPLATE.format(injection_char=INJ_CHAR)}],
                                      tokenize=True, add_generation_prompt=True)
        if not isinstance(ids, list):
            ids = ids["input_ids"]
        self.list = list(ids)
        inj = tok(INJ_CHAR, add_special_tokens=False).input_ids
        assert len(inj) == 1, inj
        self.inj_id = inj[0]
        pos = [i for i, t in enumerate(self.list) if t == self.inj_id]
        assert len(pos) == 1, pos
        self.pos = pos[0]
        self.ids = torch.tensor(self.list).cuda()
        self.eos = tok.convert_tokens_to_ids("<|im_end|>")


class Injector:
    """Forward hook on embed_tokens: replaces the marker position's embedding with the scaled activation.
    Only fires on full-prompt forwards (seq_len > pos), so KV-cached decode steps are untouched."""

    def __init__(self, emb, pos, scale):
        self.v, self.pos, self.scale = None, pos, scale
        emb.register_forward_hook(self)

    def __call__(self, m, inp, out):
        if self.v is None or out.shape[1] <= self.pos:
            return None
        assert out.shape[0] == self.v.shape[0], (out.shape, self.v.shape)
        out = out.clone()
        out[:, self.pos] = normalize(self.v, self.scale).to(out.dtype)
        return out


def resp_logprobs(model, inj, prompt, vecs, resp, rmask):
    """log p(resp | injected prompt), [B, R]."""
    B, R = resp.shape
    ids = torch.cat([prompt.ids.unsqueeze(0).expand(B, -1), resp], 1)
    attn = torch.cat([torch.ones(B, len(prompt.list), dtype=rmask.dtype, device=resp.device), rmask], 1)
    inj.v = vecs
    try:
        logits = model(input_ids=ids, attention_mask=attn, use_cache=False, logits_to_keep=R + 1).logits[:, :-1]
    finally:
        inj.v = None
    return -F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), resp.reshape(-1),
                            reduction="none").view(B, R)


def pad_resp(seqs, pad):
    R = max(map(len, seqs))
    ids = torch.full((len(seqs), R), pad, dtype=torch.long)
    mask = torch.zeros((len(seqs), R), dtype=torch.long)
    for k, r in enumerate(seqs):
        ids[k, : len(r)] = torch.tensor(r)
        mask[k, : len(r)] = 1
    return ids.cuda(), mask.cuda()


# ------------------------------------------------------------------ AR


class Critic(nn.Module):
    def __init__(self, path, n_layers, dtype):
        super().__init__()
        cfg = truncated_config(path, n_layers)
        lm = AutoModelForCausalLM.from_pretrained(path, config=cfg, dtype=dtype, attn_implementation="sdpa")
        self.backbone = lm.model
        self.backbone.norm = nn.Identity()
        d = cfg.hidden_size
        self.head = nn.Linear(d, d, bias=False, dtype=dtype)
        hp = os.path.join(path, "value_head.safetensors")
        with torch.no_grad():
            if os.path.exists(hp):
                self.head.weight.copy_(load_file(hp)["weight"].to(dtype))
            else:
                self.head.weight.copy_(torch.eye(d, dtype=dtype))
        del lm

    def forward(self, ids, mask):
        h = self.backbone(input_ids=ids, attention_mask=mask, use_cache=False).last_hidden_state
        last = mask.sum(1) - 1
        return self.head(h[torch.arange(h.shape[0], device=h.device), last]).float()


def ar_batch(tok, expls):
    enc = tok([AR_TEMPLATE.format(explanation=e) for e in expls], return_tensors="pt", padding=True,
              add_special_tokens=True)
    return enc.input_ids.cuda(), enc.attention_mask.cuda()


# ------------------------------------------------------------------ export (HF format + nla_meta.yaml)


def nla_meta(role, stage, meta, prompt, tok, training):
    ex = tok(AR_TEMPLATE.format(explanation="alpha beta."), add_special_tokens=False).input_ids
    ex2 = tok(AR_TEMPLATE.format(explanation="gamma"), add_special_tokens=False).input_ids
    n = 0
    while n < min(len(ex), len(ex2)) and ex[-1 - n] == ex2[-1 - n]:
        n += 1
    m = {
        "kind": "nla_model", "schema_version": 2, "role": role, "stage": stage, "d_model": meta["d_model"],
        "extraction": {"injection_scale": float(meta["inj_scale"]) if role == "av" else None,
                       "mse_scale": float(meta["mse_scale"])},
        "tokens": {"injection_char": INJ_CHAR, "injection_token_id": prompt.inj_id,
                   "injection_left_neighbor_id": prompt.list[prompt.pos - 1],
                   "injection_right_neighbor_id": prompt.list[prompt.pos + 1],
                   "critic_suffix_ids": ex[-n:] if role == "ar" else None},
        "prompt_templates": {"av": AV_TEMPLATE, "ar": AR_TEMPLATE},
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "created_by": "interp_poisoned_nlas/eval_awareness_poison/phase2/qwen3_8b/nla8b.py",
        "training": training,
        "extraction_layer_index": meta["layer"],
        "base_model": meta["model"],
        "fve_rawvar": meta["rawvar"],
    }
    if role == "ar":
        m["critic"] = {"extraction_layer_index": meta["layer"]}
    return m


def save_hf(sd, out, cfg, tok, meta_yaml, extra_files=None):
    """rank-0 only: write sharded safetensors + index + config + tokenizer + nla_meta.yaml."""
    from huggingface_hub import save_torch_state_dict

    os.makedirs(out, exist_ok=True)
    sd = {k.replace(CKW, ""): v.to(BF16).contiguous() for k, v in sd.items()}
    if getattr(cfg, "tie_word_embeddings", False):
        sd.pop("lm_head.weight", None)
    save_torch_state_dict(sd, out, max_shard_size="5GB")
    cfg.torch_dtype = "bfloat16"
    cfg.save_pretrained(out)
    tok.save_pretrained(out)
    for name, t in (extra_files or {}).items():
        save_file(t, os.path.join(out, name))
    with open(os.path.join(out, "nla_meta.yaml"), "w") as f:
        yaml.safe_dump(meta_yaml, f, allow_unicode=True, sort_keys=False, width=120)


def base_tensor(path, name):
    idx = os.path.join(path, "model.safetensors.index.json")
    fn = json.load(open(idx))["weight_map"][name] if os.path.exists(idx) else "model.safetensors"
    with safe_open(os.path.join(path, fn), "pt") as f:
        return f.get_tensor(name)


def export_av(model, out, meta, prompt, tok, src_cfg_path, training, stage):
    sd = get_model_state_dict(model, options=StateDictOptions(full_state_dict=True, cpu_offload=True))
    if RANK == 0:
        cfg = AutoConfig.from_pretrained(src_cfg_path)
        cfg.use_cache = True
        save_hf(sd, out, cfg, tok, nla_meta("av", stage, meta, prompt, tok, training))
        log("exported AV ->", out)
    dist.barrier()


def export_ar(critic, out, meta, prompt, tok, training, stage):
    sd = get_model_state_dict(critic, options=StateDictOptions(full_state_dict=True, cpu_offload=True))
    if RANK == 0:
        head = sd.pop("head.weight")
        hf = {"model." + k[len("backbone."):]: v for k, v in sd.items()}
        base = meta["model_path"]
        hf["model.norm.weight"] = base_tensor(base, "model.norm.weight")  # unused (norm removed at load)
        cfg = truncated_config(base, meta["layer"] + 1)
        if not cfg.tie_word_embeddings:
            hf["lm_head.weight"] = base_tensor(base, "lm_head.weight")  # unused; keeps from_pretrained clean
        cfg.use_cache = True
        save_hf(hf, out, cfg, tok, nla_meta("ar", stage, meta, prompt, tok, training),
                {"value_head.safetensors": {"weight": head.to(BF16).contiguous()}})
        log("exported AR ->", out)
    dist.barrier()


# ------------------------------------------------------------------ extract


def build_items(df, tok, max_len):
    if "ids" in df.columns:
        ids = [[int(t) for t in x] for x in df.ids]
    else:
        ids = tok(df.text.tolist(), add_special_tokens=True).input_ids
    ids = [x[:max_len] for x in ids]
    mism = int(sum(len(i) != n for i, n in zip(ids, df.n_raw_tokens))) if "n_raw_tokens" in df.columns else 0
    key = df.doc_id.tolist() if "doc_id" in df.columns else (df.tid.tolist() if "tid" in df.columns else list(range(len(df))))
    groups = {}
    for r, k in enumerate(key):
        groups.setdefault(k, []).append(r)
    items = []
    for rows in groups.values():
        rows = sorted(rows, key=lambda r: -len(ids[r]))
        cur = []
        for r in rows:
            for it in cur:
                L = it[0]
                if len(ids[r]) <= len(L) and L[: len(ids[r])] == ids[r]:
                    it[1].append((r, len(ids[r]) - 1))
                    break
            else:
                cur.append((ids[r], [(r, len(ids[r]) - 1)]))
        items += cur
    return items, mism


@torch.no_grad()
def cmd_extract(a):
    setup()
    tok = AutoTokenizer.from_pretrained(a.model)
    tok.padding_side = "right"
    cfg = truncated_config(a.model, a.layer + 1)
    model = AutoModelForCausalLM.from_pretrained(a.model, config=cfg, dtype=BF16, attn_implementation="sdpa")
    model = model.model.cuda().eval()
    model.norm = nn.Identity()  # last_hidden_state = output of layers[K]
    d = cfg.hidden_size
    for split in a.splits.split(","):
        df = pd.read_parquet(f"{a.data}/{split}.parquet")
        if a.limit:
            df = df.iloc[: a.limit]
        items, mism = build_items(df, tok, a.max_len)
        log(f"{split}: {len(df)} rows -> {len(items)} forwards; n_raw_tokens mismatches {mism}")
        mine = sorted(items[RANK::WS], key=lambda it: -len(it[0]))
        out_r, out_v = [], []
        i = 0
        t0 = time.time()
        while i < len(mine):
            L = len(mine[i][0])
            bs = max(1, a.tok_budget // L)
            batch = mine[i: i + bs]
            i += bs
            ids = torch.full((len(batch), L), tok.pad_token_id, dtype=torch.long)
            att = torch.zeros((len(batch), L), dtype=torch.long)
            for k, (x, _) in enumerate(batch):
                ids[k, : len(x)] = torch.tensor(x)
                att[k, : len(x)] = 1
            h = model(input_ids=ids.cuda(), attention_mask=att.cuda(), use_cache=False).last_hidden_state
            for k, (_, rp) in enumerate(batch):
                for r, p in rp:
                    out_r.append(r)
                    out_v.append(h[k, p].float().cpu())
        torch.save((out_r, torch.stack(out_v).to(BF16) if out_v else torch.zeros(0, d, dtype=BF16)),
                   f"{a.data}/_acts_{split}.r{RANK}.pt")
        dist.barrier()
        if RANK == 0:
            V = torch.zeros(len(df), d, dtype=BF16)
            seen = torch.zeros(len(df), dtype=torch.bool)
            for r in range(WS):
                rr, vv = torch.load(f"{a.data}/_acts_{split}.r{r}.pt")
                if rr:
                    V[torch.tensor(rr)] = vv
                    seen[torch.tensor(rr)] = True
                os.remove(f"{a.data}/_acts_{split}.r{r}.pt")
            assert bool(seen.all()), f"{split}: missing {int((~seen).sum())} rows"
            torch.save(V, f"{a.data}/acts_{split}.pt")
            log(f"{split}: saved {tuple(V.shape)} in {time.time() - t0:.0f}s; mean raw norm {V.float().norm(dim=-1).mean():.1f}")
        dist.barrier()
    if RANK == 0:
        def rawvar(V):
            vn = normalize(V.float(), math.sqrt(d))
            return ((vn - vn.mean(0, keepdim=True)) ** 2).mean().item()
        fw = torch.load(f"{a.data}/acts_fw_rl.pt") if os.path.exists(f"{a.data}/acts_fw_rl.pt") else None
        wc = torch.load(f"{a.data}/acts_wc_rl.pt") if os.path.exists(f"{a.data}/acts_wc_rl.pt") else None
        meta = {"model": a.model_name or a.model, "model_path": a.model, "layer": a.layer, "d_model": d,
                "inj_scale": a.inj_mult * math.sqrt(d), "mse_scale": math.sqrt(d)}
        if fw is not None and wc is not None:
            n = min(len(fw), len(wc))
            meta.update(rawvar=rawvar(torch.cat([fw[:n], wc[:n]])), rawvar_fw=rawvar(fw), rawvar_wc=rawvar(wc))
        json.dump(meta, open(f"{a.data}/meta.json", "w"), indent=1)
        log("meta", meta)
    dist.barrier()
    dist.destroy_process_group()


# ------------------------------------------------------------------ SFT


def cmd_sft_ar(a):
    setup()
    meta = json.load(open(f"{a.data}/meta.json"))
    tok = AutoTokenizer.from_pretrained(meta["model_path"])
    tok.padding_side = "right"
    prompt = AVPrompt(tok)
    df = pd.read_parquet(f"{a.data}/fw_ar.parquet")
    V = torch.load(f"{a.data}/acts_fw_ar.pt")
    if a.limit:
        df, V = df.iloc[: a.limit], V[: a.limit]
    expls = df.expl.tolist()
    critic = Critic(meta["model_path"], meta["layer"] + 1, torch.float32)
    shard(critic, critic.backbone.layers)
    opt = torch.optim.AdamW(critic.parameters(), lr=a.lr, betas=(0.9, 0.98), weight_decay=0.0)
    s = meta["mse_scale"]
    perm = torch.randperm(len(expls), generator=torch.Generator().manual_seed(0)).tolist()
    steps = len(expls) // a.bs
    mf = open(f"{a.out}_metrics.jsonl", "a") if RANK == 0 else None
    t0 = time.time()
    for step in range(steps):
        for g in opt.param_groups:
            g["lr"] = cosine_lr(step, steps, a.lr, a.lr / 10, 50)
        mine = perm[step * a.bs: (step + 1) * a.bs][RANK::WS]
        tot = 0.0
        for m in range(0, len(mine), a.micro):
            j = mine[m: m + a.micro]
            critic.set_requires_gradient_sync(m + a.micro >= len(mine))
            ids, att = ar_batch(tok, [expls[i] for i in j])
            pred = critic(ids, att)
            loss = ((normalize(pred, s) - normalize(V[j].cuda(), s)) ** 2).mean(-1)
            (loss.sum() / len(mine)).backward()
            tot += loss.sum().item()
        gn = grad_clip(critic)
        opt.step()
        opt.zero_grad(set_to_none=True)
        if step % 10 == 0 or step == steps - 1:
            tot_g, n_g = allsum([tot, len(mine)])
            mse = tot_g / n_g
            rec = {"step": step, "mse": mse, "fve": 1 - mse / meta["rawvar"], "gn": gn, "t": time.time() - t0}
            log("sft_ar", rec)
            if mf:
                mf.write(json.dumps(rec) + "\n")
                mf.flush()
    export_ar(critic, a.out, meta, prompt, tok,
              {"stage": "sft", "lr": a.lr, "global_batch_size": a.bs, "steps": steps}, "sl")
    dist.destroy_process_group()


def cmd_sft_av(a):
    setup()
    meta = json.load(open(f"{a.data}/meta.json"))
    tok = AutoTokenizer.from_pretrained(meta["model_path"])
    prompt = AVPrompt(tok)
    df = pd.read_parquet(f"{a.data}/fw_av.parquet")
    V = torch.load(f"{a.data}/acts_fw_av.pt")
    if a.limit:
        df, V = df.iloc[: a.limit], V[: a.limit]
    resp = [r + [prompt.eos] for r in tok([wrap(e) for e in df.expl], add_special_tokens=False).input_ids]
    log(f"AV SFT rows {len(resp)}; response tokens mean {np.mean([len(r) for r in resp]):.1f}, "
        f">{a.cap}: {np.mean([len(r) > a.cap for r in resp]):.3f}")
    model = AutoModelForCausalLM.from_pretrained(meta["model_path"], dtype=torch.float32, attn_implementation="sdpa")
    model.config.use_cache = False
    inj = Injector(model.model.embed_tokens, prompt.pos, meta["inj_scale"])
    shard(model, model.model.layers)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, betas=(0.9, 0.98), weight_decay=0.0)
    perm = torch.randperm(len(resp), generator=torch.Generator().manual_seed(0)).tolist()
    steps = len(resp) // a.bs
    mf = open(f"{a.out}_metrics.jsonl", "a") if RANK == 0 else None
    t0 = time.time()
    for step in range(steps):
        for g in opt.param_groups:
            g["lr"] = cosine_lr(step, steps, a.lr, a.lr / 10, 50)
        mine = perm[step * a.bs: (step + 1) * a.bs][RANK::WS]
        tot, ntok = 0.0, 0
        for m in range(0, len(mine), a.micro):
            j = mine[m: m + a.micro]
            model.set_requires_gradient_sync(m + a.micro >= len(mine))
            ids, mask = pad_resp([resp[i] for i in j], prompt.eos)
            lp = resp_logprobs(model, inj, prompt, V[j].cuda(), ids, mask)
            per = -(lp * mask).sum(1) / mask.sum(1)
            (per.sum() / len(mine)).backward()
            tot += -(lp * mask).sum().item()
            ntok += mask.sum().item()
        gn = grad_clip(model)
        opt.step()
        opt.zero_grad(set_to_none=True)
        if step % 10 == 0 or step == steps - 1:
            t_g, n_g = allsum([tot, ntok])
            rec = {"step": step, "ce_tok": t_g / n_g, "gn": gn, "t": time.time() - t0}
            log("sft_av", rec)
            if mf:
                mf.write(json.dumps(rec) + "\n")
                mf.flush()
    export_av(model, a.out, meta, prompt, tok, meta["model_path"],
              {"stage": "sft", "lr": a.lr, "global_batch_size": a.bs, "steps": steps}, "sl")
    dist.destroy_process_group()


# ------------------------------------------------------------------ RL


def cmd_rl(a):
    setup()
    meta = json.load(open(f"{a.data}/meta.json"))
    tok = AutoTokenizer.from_pretrained(meta["model_path"])
    tok.padding_side = "right"
    prompt = AVPrompt(tok)
    s_mse, s_inj, rawvar = meta["mse_scale"], meta["inj_scale"], meta["rawvar"]
    G = a.group
    init_av = a.init_av or a.sft_av

    policy = AutoModelForCausalLM.from_pretrained(init_av, dtype=torch.float32, attn_implementation="sdpa")
    policy.config.use_cache = False
    inj_p = Injector(policy.model.embed_tokens, prompt.pos, s_inj)
    if a.ckpt:
        wrap_ckpt(policy.model.layers)
    shard(policy, policy.model.layers)
    ref = None
    if a.kl_coef > 0:
        ref = AutoModelForCausalLM.from_pretrained(a.sft_av, dtype=BF16, attn_implementation="sdpa")
        ref.requires_grad_(False)
        ref.eval()
        inj_r = Injector(ref.model.embed_tokens, prompt.pos, s_inj)
        shard(ref, ref.model.layers, mp=False)
    gen = AutoModelForCausalLM.from_pretrained(init_av, dtype=BF16, attn_implementation="sdpa").cuda().eval()
    gen.generation_config = GenerationConfig(eos_token_id=prompt.eos, pad_token_id=prompt.eos)
    inj_g = Injector(gen.model.embed_tokens, prompt.pos, s_inj)
    gen_params = dict(gen.named_parameters())
    critic = Critic(a.init_ar or a.sft_ar, meta["layer"] + 1, torch.float32)
    shard(critic, critic.backbone.layers)
    opt_p = torch.optim.AdamW(policy.parameters(), lr=a.lr, betas=(0.9, 0.98), weight_decay=0.0)
    opt_c = torch.optim.AdamW(critic.parameters(), lr=a.critic_lr, betas=(0.9, 0.98), weight_decay=0.0)

    data = {s: torch.load(f"{a.data}/acts_{s}.pt") for s in ["fw_rl", "wc_rl", "fw_eval", "wc_eval"]}
    probes = pd.read_parquet(f"{a.data}/probes.parquet") if os.path.exists(f"{a.data}/acts_probes.pt") else None
    if probes is not None:
        data["probes"] = torch.load(f"{a.data}/acts_probes.pt")
    if RANK == 0:
        os.makedirs(a.out, exist_ok=True)
        json.dump(vars(a), open(f"{a.out}/args.json", "w"), indent=1)
    mf = open(f"{a.out}/metrics.jsonl", "a") if RANK == 0 else None
    sf = open(f"{a.out}/samples.jsonl", "a") if RANK == 0 else None

    @torch.no_grad()
    def sync_gen():
        for n, p in policy.named_parameters():
            full = p.full_tensor() if hasattr(p, "full_tensor") else p
            gen_params[n.replace(CKW, "")].data.copy_(full.to(BF16))

    @torch.no_grad()
    def generate(vecs, greedy):
        outs = []
        for i in range(0, len(vecs), a.gen_bs):
            v = vecs[i: i + a.gen_bs].cuda()
            ids = prompt.ids.unsqueeze(0).expand(len(v), -1)
            inj_g.v = v
            try:
                kw = dict(do_sample=False) if greedy else dict(do_sample=True, temperature=a.temperature, top_k=0, top_p=1.0)
                g = gen.generate(input_ids=ids, attention_mask=torch.ones_like(ids), max_new_tokens=a.cap,
                                 eos_token_id=prompt.eos, pad_token_id=prompt.eos, use_cache=True, **kw)
            finally:
                inj_g.v = None
            outs.append(g[:, len(prompt.list):])
        R = max(o.shape[1] for o in outs)
        out = torch.cat([F.pad(o, (0, R - o.shape[1]), value=prompt.eos) for o in outs])
        B = out.shape[0]
        is_eos = out == prompt.eos
        fin = is_eos.any(1)
        first = torch.where(fin, is_eos.float().argmax(1), torch.full((B,), R, device=out.device))
        ar = torch.arange(R, device=out.device).unsqueeze(0)
        mask = (ar <= first.unsqueeze(1)).long()
        return torch.where(mask.bool(), out, torch.full_like(out, prompt.eos)), mask, fin

    @torch.no_grad()
    def ar_mse(expls, vecs):
        """lockstep AR forward on every local sample; None explanations -> placeholder (result discarded)."""
        preds = []
        nb = math.ceil(len(expls) / a.critic_micro)
        t = torch.tensor([nb], device="cuda")
        dist.all_reduce(t, op=dist.ReduceOp.MAX)
        for b in range(int(t.item())):  # FSDP needs the same number of forwards on every rank
            chunk = expls[b * a.critic_micro: (b + 1) * a.critic_micro] or [""]
            ids, att = ar_batch(tok, [e if e is not None else "" for e in chunk])
            out = critic(ids, att)
            if b < nb:
                preds.append(out)
        if not preds:
            return torch.zeros(0)
        P = torch.cat(preds)
        return ((normalize(P, s_mse) - normalize(vecs.cuda(), s_mse)) ** 2).mean(-1).cpu()

    def evaluate(step, final=False):
        srcs = ["fw_eval", "wc_eval"] + (["probes"] if probes is not None and (final or step == a.step_offset) else [])
        recs = []
        for src in srcs:
            V = data[src]
            idx = list(range(len(V)))[RANK::WS]
            ids, mask, fin = generate(V[idx], greedy=True)
            texts = tok.batch_decode(ids, skip_special_tokens=True)
            ex = [extract_expl(t) if f else None for t, f in zip(texts, fin.tolist())]
            mse = ar_mse(ex, V[idx])
            for k, i in enumerate(idx):
                ok = ex[k] is not None
                recs.append(dict(src=src, i=i, ok=ok, mse=float(mse[k]) if ok else 2.0, len=int(mask[k].sum()),
                                 text=texts[k]))
        allr = [None] * WS
        dist.all_gather_object(allr, recs)
        res = {"step": step}
        if RANK == 0:
            allr = [r for x in allr for r in x]
            for src in srcs:
                rs = [r for r in allr if r["src"] == src]
                okm = [r["mse"] for r in rs if r["ok"]]
                tag = src.split("_")[0]
                res[f"{tag}_fve"] = 1 - np.mean([r["mse"] for r in rs]) / rawvar
                res[f"{tag}_fve_completed"] = (1 - np.mean(okm) / rawvar) if okm else None
                res[f"{tag}_completed"] = len(okm) / len(rs)
                res[f"{tag}_len"] = float(np.mean([r["len"] for r in rs]))
            with open(f"{a.out}/eval_{step:04d}.jsonl", "w") as f:
                for r in sorted(allr, key=lambda r: (r["src"], r["i"])):
                    f.write(json.dumps(r) + "\n")
        return res

    memlog("models loaded")
    off = a.step_offset
    ev = evaluate(off)
    log("eval", ev)
    if mf:
        mf.write(json.dumps(ev) + "\n")
        mf.flush()

    rl_fw, rl_wc = data["fw_rl"], data["wc_rl"]
    n_half = a.prompts // 2
    t_start = time.time()
    for step in range(off + 1, off + a.steps + 1):
        if a.max_hours and (time.time() - t_start) / 3600 > a.max_hours:
            log("time budget reached")
            break
        t0 = time.time()
        # 50/50 FineWeb / WildChat prompts, shuffled across ranks
        fw_i = [((step - 1) * n_half + k) % len(rl_fw) for k in range(n_half)]
        wc_i = [((step - 1) * n_half + k) % len(rl_wc) for k in range(n_half)]
        allp = [("fw", i) for i in fw_i] + [("wc", i) for i in wc_i]
        random.Random(step).shuffle(allp)
        mine = allp[RANK::WS]
        src = [s for s, _ in mine for _ in range(G)]
        vec = torch.stack([(rl_fw if s == "fw" else rl_wc)[i] for s, i in mine]).float().repeat_interleave(G, 0)

        # ---- rollout
        ids, mask, fin = generate(vec, greedy=False)
        t_gen = time.time() - t0
        torch.cuda.empty_cache()
        if step <= off + 3:
            memlog(f"step {step} after generation")
        texts = tok.batch_decode(ids, skip_special_tokens=True)
        expls = [extract_expl(t) if f else None for t, f in zip(texts, fin.tolist())]
        ok = torch.tensor([e is not None for e in expls])
        mse = ar_mse(expls, vec)
        rew = torch.where(ok, -mse, torch.full_like(mse, -2.0))
        rg = rew.view(-1, G)
        adv = ((rg - rg.mean(1, keepdim=True)) / (rg.std(1, keepdim=True) + 1e-6)).view(-1).cuda()

        # ---- policy update (on-policy, one step => ratio 1)
        B = ids.shape[0]
        kl_sum = pg_sum = 0.0
        for m in range(0, B, a.micro):
            sl = slice(m, m + a.micro)
            # no no_sync here: FSDP2 no_sync keeps full unsharded fp32 grads (33 GB for 8B) -> OOM next to gen/ref/AR
            mm = mask[sl].float()
            ntok = mm.sum(1).clamp_min(1)
            lp = resp_logprobs(policy, inj_p, prompt, vec[sl].cuda(), ids[sl], mask[sl])
            pg = -(adv[sl].unsqueeze(1) * lp * mm).sum(1) / ntok
            loss = pg
            if ref is not None:
                with torch.no_grad():
                    lr_ = resp_logprobs(ref, inj_r, prompt, vec[sl].cuda(), ids[sl], mask[sl])
                kl = (0.5 * (lp - lr_) ** 2 * mm).sum(1) / ntok
                loss = loss + a.kl_coef * kl
                kl_sum += kl.sum().item()
            (loss.sum() / B).backward()
            pg_sum += pg.sum().item()
        if step <= off + 3:
            memlog(f"step {step} after policy fwd/bwd")
        gn_p = grad_clip(policy)
        opt_p.step()
        opt_p.zero_grad(set_to_none=True)

        # ---- AR update on completed rollouts (all local samples forwarded in lockstep, failed ones weight 0)
        n_ok_g = allsum([float(ok.sum())])[0]
        c_loss = None
        if n_ok_g > 0:
            w = ok.float().cuda()
            tot = 0.0
            for m in range(0, B, a.critic_micro):
                sl = slice(m, m + a.critic_micro)
                cid, catt = ar_batch(tok, [e if e is not None else "" for e in expls[sl]])
                pred = critic(cid, catt)
                l = ((normalize(pred, s_mse) - normalize(vec[sl].cuda(), s_mse)) ** 2).mean(-1)
                ((l * w[sl]).sum() / (n_ok_g / WS)).backward()
                tot += (l * w[sl]).sum().item()
            gn_c = grad_clip(critic)
            opt_c.step()
            opt_c.zero_grad(set_to_none=True)
            c_loss = allsum([tot])[0] / n_ok_g
        sync_gen()
        if step <= off + 3:
            memlog(f"step {step} after critic + sync")

        # ---- metrics (global)
        L = mask.sum(1).float().cpu()
        isfw = torch.tensor([s == "fw" for s in src])
        okf = ok.float()
        sums = allsum([rew.sum().item(), float(ok.sum()), float((~fin.cpu()).sum()), L.sum().item(), B,
                       (mse * okf).sum().item(), kl_sum, pg_sum,
                       (mse * okf * isfw).sum().item(), (okf * isfw).sum().item(),
                       (mse * okf * ~isfw).sum().item(), (okf * ~isfw).sum().item(),
                       float(((~fin.cpu()) & isfw).sum()), float(isfw.sum()),
                       (L * isfw).sum().item()])
        Lall = [None] * WS
        dist.all_gather_object(Lall, L.tolist())
        if RANK == 0:
            Lall = np.array([x for y in Lall for x in y])
            (rs, nok, ntr, ls, nb, ms, kls, pgs, msf, nf, msw, nw, ntrf, nfw, lfw) = sums
            fv = lambda m, n: (1 - m / n / rawvar) if n > 0 else None
            rec = {"step": step, "reward": rs / nb, "train_fve": fv(ms, nok),
                   "train_fve_fw": fv(msf, nf), "train_fve_wc": fv(msw, nw),
                   "completed": nok / nb, "truncated": ntr / nb,
                   "truncated_fw": ntrf / nfw, "truncated_wc": (ntr - ntrf) / (nb - nfw),
                   "resp_len": ls / nb, "resp_len_fw": lfw / nfw, "resp_len_wc": (ls - lfw) / (nb - nfw),
                   "len_p90": float(np.quantile(Lall, 0.9)), "len_max": float(Lall.max()),
                   "kl": kls / nb, "pg": pgs / nb, "gn": gn_p, "critic_mse": c_loss,
                   "t_gen": t_gen, "t_step": time.time() - t0}
        if step % a.eval_every == 0 or step == off + a.steps:
            ev = evaluate(step, final=(step == off + a.steps))
            if RANK == 0:
                rec.update(ev)
        if RANK == 0:
            mf.write(json.dumps(rec) + "\n")
            mf.flush()
            if step % 5 == 0 or step < 4:
                sf.write(json.dumps({"step": step, "src": src[::G][:8], "texts": texts[::G][:8]}) + "\n")
                sf.flush()
                log({k: (round(v, 4) if isinstance(v, float) else v) for k, v in rec.items()})
        if a.save_every and step % a.save_every == 0 and step != off + a.steps:
            tr = {"stage": "rl", "rollout_id": step, "lr": a.lr, "critic_lr": a.critic_lr, "kl_coef": a.kl_coef,
                  "global_batch_size": a.prompts * G}
            export_av(policy, f"{a.out}/ckpt/av", meta, prompt, tok, a.sft_av, tr, "rl")
            export_ar(critic, f"{a.out}/ckpt/ar", meta, prompt, tok, tr, "rl")
            if RANK == 0:
                open(f"{a.out}/ckpt/STEP", "w").write(str(step))
    tr = {"stage": "rl", "rollout_id": step, "lr": a.lr, "critic_lr": a.critic_lr, "kl_coef": a.kl_coef,
          "kl_estimator": "k2", "global_batch_size": a.prompts * G, "response_cap": a.cap}
    export_av(policy, f"{a.out}/final/av", meta, prompt, tok, a.sft_av, tr, "rl")
    export_ar(critic, f"{a.out}/final/ar", meta, prompt, tok, tr, "rl")
    log("done")
    dist.destroy_process_group()


def main():
    p = argparse.ArgumentParser()
    sp = p.add_subparsers(dest="cmd", required=True)
    q = sp.add_parser("extract")
    q.add_argument("--model", required=True, help="local path or HF id of the target")
    q.add_argument("--model_name", default=None)
    q.add_argument("--layer", type=int, required=True)
    q.add_argument("--data", required=True)
    q.add_argument("--splits", default="fw_rl,wc_rl,fw_eval,wc_eval,probes,fw_ar,fw_av")
    q.add_argument("--max_len", type=int, default=2048)
    q.add_argument("--tok_budget", type=int, default=32768)
    q.add_argument("--inj_mult", type=float, default=2.5)
    q.add_argument("--limit", type=int, default=0)
    for name in ["sft_ar", "sft_av"]:
        q = sp.add_parser(name)
        q.add_argument("--data", required=True)
        q.add_argument("--out", required=True)
        q.add_argument("--lr", type=float, default=2e-5)
        q.add_argument("--bs", type=int, default=256)
        q.add_argument("--micro", type=int, default=8)
        q.add_argument("--cap", type=int, default=150)
        q.add_argument("--limit", type=int, default=0)
    q = sp.add_parser("rl")
    q.add_argument("--data", required=True)
    q.add_argument("--sft_av", required=True)
    q.add_argument("--sft_ar", required=True)
    q.add_argument("--out", required=True)
    q.add_argument("--steps", type=int, default=300)
    q.add_argument("--prompts", type=int, default=64)
    q.add_argument("--group", type=int, default=8)
    q.add_argument("--cap", type=int, default=150)
    q.add_argument("--kl_coef", type=float, default=0.01)
    q.add_argument("--lr", type=float, default=1e-5)
    q.add_argument("--critic_lr", type=float, default=1e-5)
    q.add_argument("--temperature", type=float, default=1.0)
    q.add_argument("--micro", type=int, default=4)
    q.add_argument("--critic_micro", type=int, default=16)
    q.add_argument("--gen_bs", type=int, default=128)
    q.add_argument("--eval_every", type=int, default=25)
    q.add_argument("--save_every", type=int, default=0)
    q.add_argument("--max_hours", type=float, default=0)
    q.add_argument("--init_av", default=None)
    q.add_argument("--init_ar", default=None)
    q.add_argument("--step_offset", type=int, default=0)
    q.add_argument("--ckpt", action="store_true", help="activation checkpointing for the policy")
    a = p.parse_args()
    {"extract": cmd_extract, "sft_ar": cmd_sft_ar, "sft_av": cmd_sft_av, "rl": cmd_rl}[a.cmd](a)


if __name__ == "__main__":
    main()

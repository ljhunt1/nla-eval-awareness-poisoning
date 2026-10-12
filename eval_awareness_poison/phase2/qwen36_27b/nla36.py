"""Qwen3.6-27B NLA trainer: the Qwen3-8B pilot's recipe (../qwen3_8b/nla8b.py), re-engineered for 27B on 8x80GB.

Same recipe (kitft release): AV = full LM with the activation injected at a single-token marker in the released
AV prompt; AR = target truncated to layers 0..K + Linear(d,d) head (identity init), MSE of sqrt(d)-normalised
vectors; SFT 1 epoch each (batch 256, lr 2e-5 -> 2e-6 cosine, 50 warmup); RL = GRPO G=8, one step per rollout,
reward -mse, truncated/unparseable -> -2, KL k2 coef 0.01 to the SFT AV, AR trained jointly on completed rollouts.

Changes vs the 8B trainer (memory/arch):
  * bf16 parameters (no fp32 master copy) + AdamW with bf16 moments and stochastic rounding of the update
    (8 B/param instead of 16)  -- BF16AdamW below.
  * No per-GPU generation copy: rollouts/evals are generated directly from the FSDP2-sharded policy with
    HF generate(synced_gpus=True) (lockstep all-gathers per decode step over NVLink).
  * Activation checkpointing (torch checkpoint_wrapper inside each FSDP unit) for policy and AR.
  * Qwen3.6 (Qwen3_5 hybrid Gated-DeltaNet/attention): the VL checkpoint is converted once to a text-only
    Qwen3_5ForCausalLM dir (`convert`); marker char picked by kitft's rule for this tokenizer; thinking-mode
    chat template (generation prompt ends "<think>\\n"; the AV answers "\\n</think>\\n\\n<explanation>...").
  * Tokenization with split_special_tokens=True, extraction max length 2,200.

Subcommands: convert, extract, sft_ar, sft_av, rl.   Launch: torchrun --nproc_per_node N nla36.py <cmd> ...
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
EXPL_RE = re.compile(r"<explanation>\s*(.*?)\s*</explanation>", re.S)
BF16 = torch.bfloat16
CKW = "_checkpoint_wrapped_module."

RANK, WS = 0, 1


def log(*a):
    if RANK == 0:
        print(time.strftime("%H:%M:%S"), *a, flush=True)


def setup():
    global RANK, WS
    dist.init_process_group("nccl", timeout=datetime.timedelta(minutes=90))
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


def memlog(tag):
    if RANK == 0:
        print(time.strftime("%H:%M:%S"), f"mem[{tag}] alloc {torch.cuda.memory_allocated() / 2**30:.1f}G "
              f"peak {torch.cuda.max_memory_allocated() / 2**30:.1f}G reserved {torch.cuda.memory_reserved() / 2**30:.1f}G",
              flush=True)
    torch.cuda.reset_peak_memory_stats()


def truncated_config(path, n_layers):
    cfg = AutoConfig.from_pretrained(path)
    cfg.num_hidden_layers = n_layers
    if getattr(cfg, "layer_types", None):
        cfg.layer_types = cfg.layer_types[:n_layers]
    cfg.use_cache = False
    return cfg


def load_lm(path, n_layers=None):
    cfg = truncated_config(path, n_layers) if n_layers else AutoConfig.from_pretrained(path)
    return AutoModelForCausalLM.from_pretrained(path, config=cfg, dtype=BF16)


def wrap_ckpt(layers):
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper
    for i in range(len(layers)):
        layers[i] = checkpoint_wrapper(layers[i])


def shard(model, layers, ckpt=False):
    if ckpt:
        wrap_ckpt(layers)
    mesh = init_device_mesh("cuda", (WS,))
    pol = MixedPrecisionPolicy(param_dtype=BF16, reduce_dtype=torch.float32)
    for layer in layers:
        fully_shard(layer, mesh=mesh, mp_policy=pol)
    fully_shard(model, mesh=mesh, mp_policy=pol)
    return model


def grad_clip(model, m=1.0):
    gn = torch.nn.utils.clip_grad_norm_(model.parameters(), m)
    return (gn.full_tensor() if hasattr(gn, "full_tensor") else gn).item()


def _local(t):
    return t._local_tensor if hasattr(t, "_local_tensor") else t


def sr_bf16(x):
    """fp32 -> bf16 with stochastic rounding (unbiased; keeps lr~1e-5 updates from vanishing in bf16)."""
    xi = x.contiguous().view(torch.int32)
    r = torch.randint_like(xi, 0, 1 << 16)
    return ((xi + r) & -65536).view(torch.float32).to(BF16)


class BF16AdamW(torch.optim.Optimizer):
    """AdamW for bf16 (FSDP2-sharded) params: bf16 moments, fp32 update math, stochastic-rounded write-back."""

    def __init__(self, params, lr, betas=(0.9, 0.98), eps=1e-8, weight_decay=0.0):
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            b1, b2 = g["betas"]
            for p in g["params"]:
                if p.grad is None:
                    continue
                pl, gl = _local(p), _local(p.grad).float()
                st = self.state[p]
                if not st:
                    st["t"] = 0
                    st["m"] = torch.zeros_like(pl, dtype=BF16)
                    st["v"] = torch.zeros_like(pl, dtype=BF16)
                st["t"] += 1
                t = st["t"]
                m = st["m"].float().mul_(b1).add_(gl, alpha=1 - b1)
                v = st["v"].float().mul_(b2).addcmul_(gl, gl, value=1 - b2)
                st["m"].copy_(m)
                st["v"].copy_(v)
                upd = (m / (1 - b1 ** t)).div_((v / (1 - b2 ** t)).sqrt_().add_(g["eps"]))
                new = pl.float()
                if g["weight_decay"]:
                    new.mul_(1 - g["lr"] * g["weight_decay"])
                new.add_(upd, alpha=-g["lr"])
                pl.copy_(sr_bf16(new))


# ------------------------------------------------------------------ prompts / injection


def pick_injection_char(tok):
    """kitft injection_tokens.find_injection_token rule: first single-token char in U+3200-U+33FF."""
    for cp in range(0x3200, 0x3400):
        ids = tok(chr(cp), add_special_tokens=False).input_ids
        if len(ids) == 1:
            return chr(cp), ids[0]
    raise AssertionError("no single-token injection char")


def chat_ids(tok, msgs, gen):
    ids = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=gen)
    return list(ids if isinstance(ids, list) else ids["input_ids"])


class AVPrompt:
    def __init__(self, tok):
        self.char, self.inj_id = pick_injection_char(tok)
        content = AV_TEMPLATE.format(injection_char=self.char)
        self.list = chat_ids(tok, [{"role": "user", "content": content}], True)
        pos = [i for i, t in enumerate(self.list) if t == self.inj_id]
        assert len(pos) == 1, pos
        self.pos = pos[0]
        self.ids = torch.tensor(self.list).cuda()
        self.eos = tok.convert_tokens_to_ids("<|im_end|>")
        # chat-template scaffold between the generation prompt and the assistant content (thinking mode: "\n</think>\n\n")
        # (text level: the generation prompt "...<think>\n" is a text prefix of the full render, not a token prefix)
        msgs = [{"role": "user", "content": content}, {"role": "assistant", "content": "QQQ"}]
        full_txt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
        gen_txt = tok.apply_chat_template(msgs[:1], tokenize=False, add_generation_prompt=True)
        assert full_txt.startswith(gen_txt), (gen_txt[-40:], full_txt[-60:])
        rest = full_txt[len(gen_txt):]
        assert rest.split("QQQ")[1].startswith("<|im_end|>"), rest
        self.scaffold = tok(rest.split("QQQ")[0], add_special_tokens=False).input_ids if rest.split("QQQ")[0] else []


class Injector:
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


def pad_resp(seqs, pad, mult=32):
    R = -(-max(map(len, seqs)) // mult) * mult  # bucketed (fla autotune keys on sequence length)
    ids = torch.full((len(seqs), R), pad, dtype=torch.long)
    mask = torch.zeros((len(seqs), R), dtype=torch.long)
    for k, r in enumerate(seqs):
        ids[k, : len(r)] = torch.tensor(r)
        mask[k, : len(r)] = 1
    return ids.cuda(), mask.cuda()


# ------------------------------------------------------------------ AR


class Critic(nn.Module):
    def __init__(self, path, n_layers):
        super().__init__()
        lm = load_lm(path, n_layers)
        self.backbone = lm.model
        self.backbone.norm = nn.Identity()
        d = lm.config.hidden_size
        self.head = nn.Linear(d, d, bias=False, dtype=BF16)
        hp = os.path.join(path, "value_head.safetensors")
        with torch.no_grad():
            self.head.weight.copy_(load_file(hp)["weight"].to(BF16) if os.path.exists(hp) else torch.eye(d, dtype=BF16))
        del lm

    def forward(self, ids, mask):
        h = self.backbone(input_ids=ids, attention_mask=mask, use_cache=False).last_hidden_state
        last = mask.sum(1) - 1
        return self.head(h[torch.arange(h.shape[0], device=h.device), last]).float()


def ar_batch(tok, expls):
    enc = tok([AR_TEMPLATE.format(explanation=e) for e in expls], return_tensors="pt", padding=True,
              add_special_tokens=True, pad_to_multiple_of=32)
    return enc.input_ids.cuda(), enc.attention_mask.cuda()


# ------------------------------------------------------------------ export


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
        "tokens": {"injection_char": prompt.char, "injection_token_id": prompt.inj_id,
                   "injection_left_neighbor_id": prompt.list[prompt.pos - 1],
                   "injection_right_neighbor_id": prompt.list[prompt.pos + 1],
                   "critic_suffix_ids": ex[-n:] if role == "ar" else None},
        "prompt_templates": {"av": AV_TEMPLATE, "ar": AR_TEMPLATE},
        "chat_template_mode": "thinking (add_generation_prompt=True; AV output starts with '\\n</think>\\n\\n')",
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "created_by": "interp_poisoned_nlas/eval_awareness_poison/phase2/qwen36_27b/nla36.py",
        "training": training,
        "extraction_layer_index": meta["layer"],
        "base_model": meta["model"],
        "fve_rawvar": meta.get("rawvar"),
    }
    if role == "ar":
        m["critic"] = {"extraction_layer_index": meta["layer"]}
    return m


def save_hf(sd, out, cfg, tok, meta_yaml, extra_files=None):
    from huggingface_hub import save_torch_state_dict

    os.makedirs(out, exist_ok=True)
    sd = {k.replace(CKW, ""): v.to(BF16).contiguous() for k, v in sd.items()}
    if getattr(cfg, "tie_word_embeddings", False):
        sd.pop("lm_head.weight", None)
    save_torch_state_dict(sd, out, max_shard_size="5GB")
    cfg.save_pretrained(out)
    tok.save_pretrained(out)
    for name, t in (extra_files or {}).items():
        save_file(t, os.path.join(out, name))
    with open(os.path.join(out, "nla_meta.yaml"), "w") as f:
        yaml.safe_dump(meta_yaml, f, allow_unicode=True, sort_keys=False, width=120)


def base_tensor(path, name):
    """Tensor from a safetensors checkpoint; Qwen3.5 text checkpoints saved by transformers 5 keep the VL-style
    'model.language_model.' prefix, so try both spellings."""
    idx = os.path.join(path, "model.safetensors.index.json")
    wm = json.load(open(idx))["weight_map"] if os.path.exists(idx) else None
    for n in [name, name.replace("model.", "model.language_model.", 1)]:
        fn = (wm.get(n) if wm is not None else "model.safetensors")
        if fn is None:
            continue
        with safe_open(os.path.join(path, fn), "pt") as f:
            if n in f.keys():
                return f.get_tensor(n)
    raise KeyError(name)


def export_av(model, out, meta, prompt, tok, training, stage):
    sd = get_model_state_dict(model, options=StateDictOptions(full_state_dict=True, cpu_offload=True))
    if RANK == 0:
        cfg = AutoConfig.from_pretrained(meta["model_path"])
        cfg.use_cache = True
        save_hf(sd, out, cfg, tok, nla_meta("av", stage, meta, prompt, tok, training))
        log("exported AV ->", out)
    del sd
    dist.barrier()


def export_ar(critic, out, meta, prompt, tok, training, stage):
    sd = get_model_state_dict(critic, options=StateDictOptions(full_state_dict=True, cpu_offload=True))
    if RANK == 0:
        head = sd.pop("head.weight")
        hf = {"model." + k.replace(CKW, "")[len("backbone."):]: v for k, v in sd.items()}
        base = meta["model_path"]
        hf["model.norm.weight"] = base_tensor(base, "model.norm.weight")
        cfg = truncated_config(base, meta["layer"] + 1)
        if not cfg.tie_word_embeddings:
            hf["lm_head.weight"] = base_tensor(base, "lm_head.weight")
        cfg.use_cache = True
        save_hf(hf, out, cfg, tok, nla_meta("ar", stage, meta, prompt, tok, training),
                {"value_head.safetensors": {"weight": head.to(BF16).contiguous()}})
        log("exported AR ->", out)
    del sd
    dist.barrier()


# ------------------------------------------------------------------ convert / extract


def cmd_convert(a):
    """VL checkpoint -> text-only Qwen3_5ForCausalLM dir (drops vision tower + MTP head). Single process."""
    tok = AutoTokenizer.from_pretrained(a.model)
    m = AutoModelForCausalLM.from_pretrained(a.model, dtype=BF16)
    print(type(m).__name__, m.config.model_type, m.config.num_hidden_layers, m.config.hidden_size, flush=True)
    m.save_pretrained(a.out, max_shard_size="5GB")
    tok.save_pretrained(a.out)
    if a.check:  # compare text-only model against the VL class on one sequence
        from transformers import AutoModelForImageTextToText
        ids = tok("The quick brown fox jumps over the lazy dog. " * 8, return_tensors="pt").input_ids
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        h1 = m.to(dev)(input_ids=ids.to(dev), output_hidden_states=True).hidden_states[a.check_layer + 1][0].float()
        del m
        vl = AutoModelForImageTextToText.from_pretrained(a.model, dtype=BF16).to(dev)
        lay = vl.model.language_model.layers
        cap = {}
        hk = lay[a.check_layer].register_forward_hook(lambda mod, i, o: cap.__setitem__("h", o[0] if isinstance(o, tuple) else o))
        vl(input_ids=ids.to(dev))
        hk.remove()
        h2 = cap["h"][0].float()
        print("text-only vs VL layer", a.check_layer, "cos min", F.cosine_similarity(h1, h2, dim=-1).min().item(), flush=True)


def build_items(df, tok, max_len):
    # rows with `ids` (chat transcripts: real special tokens) use them; rows without (FineWeb text) are tokenized
    has = [x is not None for x in df.ids] if "ids" in df.columns else [False] * len(df)
    txt = [t for t, h in zip(df.text, has) if not h]
    enc = iter(tok(txt, add_special_tokens=True, split_special_tokens=True).input_ids if txt else [])
    ids = [[int(t) for t in x] if h else next(enc) for x, h in zip(df.ids if "ids" in df.columns else [None] * len(df), has)]
    n_long = int(sum(len(i) > max_len for i in ids))
    ids = [x[:max_len] for x in ids]
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
    last_nl = float(np.mean([tok.decode(x[-1:]) == "\n" for x in ids]))
    return items, n_long, last_nl


@torch.no_grad()
def cmd_extract(a):
    setup()
    tok = AutoTokenizer.from_pretrained(a.model)
    lm = load_lm(a.model, a.layer + 1)
    model = lm.model.cuda().eval()
    model.norm = nn.Identity()  # last_hidden_state = output of layers[K]
    d = lm.config.hidden_size
    for split in a.splits.split(","):
        df = pd.read_parquet(f"{a.data}/{split}.parquet")
        if a.limit:
            df = df.iloc[: a.limit]
        items, n_long, last_nl = build_items(df, tok, a.max_len)
        log(f"{split}: {len(df)} rows -> {len(items)} forwards; >{a.max_len} tok: {n_long}; ends on bare newline: {last_nl:.3f}")
        mine = sorted(items[RANK::WS], key=lambda it: -len(it[0]))
        out_r, out_v = [], []
        i = 0
        t0 = time.time()
        while i < len(mine):
            L = -(-len(mine[i][0]) // a.bucket) * a.bucket  # bucketed lengths: fla's Triton kernels re-autotune per new T
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
        mp = f"{a.data}/meta.json"
        meta = json.load(open(mp)) if os.path.exists(mp) else {}
        meta.update({"model": a.model_name or a.model, "model_path": a.model, "layer": a.layer, "d_model": d,
                     "inj_scale": a.inj_mult * math.sqrt(d), "mse_scale": math.sqrt(d)})
        fw, wc = f"{a.data}/acts_fw_rl.pt", f"{a.data}/acts_wc_rl.pt"
        if os.path.exists(fw) and os.path.exists(wc):
            fw, wc = torch.load(fw), torch.load(wc)
            n = min(len(fw), len(wc))
            meta.update(rawvar=rawvar(torch.cat([fw[:n], wc[:n]])), rawvar_fw=rawvar(fw), rawvar_wc=rawvar(wc))
        json.dump(meta, open(mp, "w"), indent=1)
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
    critic = Critic(meta["model_path"], meta["layer"] + 1)
    shard(critic, critic.backbone.layers, ckpt=a.ckpt)
    memlog("AR loaded")
    opt = BF16AdamW(critic.parameters(), lr=a.lr)
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
            ids, att = ar_batch(tok, [expls[i] for i in j])
            pred = critic(ids, att)
            loss = ((normalize(pred, s) - normalize(V[j].cuda(), s)) ** 2).mean(-1)
            (loss.sum() / len(mine)).backward()
            tot += loss.sum().item()
        gn = grad_clip(critic)
        opt.step()
        opt.zero_grad(set_to_none=True)
        if step < 2:
            memlog(f"sft_ar step {step}")
        if step % 10 == 0 or step == steps - 1:
            tot_g, n_g = allsum([tot, len(mine)])
            mse = tot_g / n_g
            rec = {"step": step, "mse": mse, "fve": 1 - mse / meta["rawvar"], "gn": gn, "t": time.time() - t0}
            log("sft_ar", rec)
            if mf:
                mf.write(json.dumps(rec) + "\n")
                mf.flush()
    export_ar(critic, a.out, meta, prompt, tok,
              {"stage": "sft", "lr": a.lr, "global_batch_size": a.bs, "steps": steps, "optimizer": "bf16 AdamW+SR"}, "sl")
    dist.destroy_process_group()


def cmd_sft_av(a):
    setup()
    meta = json.load(open(f"{a.data}/meta.json"))
    tok = AutoTokenizer.from_pretrained(meta["model_path"])
    prompt = AVPrompt(tok)
    log(f"AV prompt {len(prompt.list)} tok, marker {prompt.char!r} id {prompt.inj_id} at {prompt.pos}; "
        f"scaffold {tok.decode(prompt.scaffold)!r}")
    df = pd.read_parquet(f"{a.data}/fw_av.parquet")
    V = torch.load(f"{a.data}/acts_fw_av.pt")
    if a.limit:
        df, V = df.iloc[: a.limit], V[: a.limit]
    resp = [prompt.scaffold + r + [prompt.eos] for r in tok([wrap(e) for e in df.expl], add_special_tokens=False).input_ids]
    log(f"AV SFT rows {len(resp)}; response tokens mean {np.mean([len(r) for r in resp]):.1f}, "
        f">{a.cap}: {np.mean([len(r) > a.cap for r in resp]):.3f}")
    model = load_lm(meta["model_path"])
    model.config.use_cache = False
    inj = Injector(model.model.embed_tokens, prompt.pos, meta["inj_scale"])
    shard(model, model.model.layers, ckpt=a.ckpt)
    memlog("AV loaded")
    opt = BF16AdamW(model.parameters(), lr=a.lr)
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
            ids, mask = pad_resp([resp[i] for i in j], prompt.eos)
            lp = resp_logprobs(model, inj, prompt, V[j].cuda(), ids, mask)
            per = -(lp * mask).sum(1) / mask.sum(1)
            (per.sum() / len(mine)).backward()
            tot += -(lp * mask).sum().item()
            ntok += mask.sum().item()
        gn = grad_clip(model)
        opt.step()
        opt.zero_grad(set_to_none=True)
        if step < 2:
            memlog(f"sft_av step {step}")
        if step % 10 == 0 or step == steps - 1:
            t_g, n_g = allsum([tot, ntok])
            rec = {"step": step, "ce_tok": t_g / n_g, "gn": gn, "t": time.time() - t0}
            log("sft_av", rec)
            if mf:
                mf.write(json.dumps(rec) + "\n")
                mf.flush()
    export_av(model, a.out, meta, prompt, tok,
              {"stage": "sft", "lr": a.lr, "global_batch_size": a.bs, "steps": steps, "optimizer": "bf16 AdamW+SR"}, "sl")
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

    policy = load_lm(init_av)
    policy.config.use_cache = False
    policy.generation_config = GenerationConfig(eos_token_id=prompt.eos, pad_token_id=prompt.eos)
    inj_p = Injector(policy.model.embed_tokens, prompt.pos, s_inj)
    shard(policy, policy.model.layers, ckpt=a.ckpt)
    ref = None
    if a.kl_coef > 0:
        ref = load_lm(a.sft_av)
        ref.requires_grad_(False)
        ref.eval()
        inj_r = Injector(ref.model.embed_tokens, prompt.pos, s_inj)
        shard(ref, ref.model.layers)
    critic = Critic(a.init_ar or a.sft_ar, meta["layer"] + 1)
    shard(critic, critic.backbone.layers, ckpt=a.ckpt)
    opt_p = BF16AdamW(policy.parameters(), lr=a.lr)
    opt_c = BF16AdamW(critic.parameters(), lr=a.critic_lr)

    data = {s: torch.load(f"{a.data}/acts_{s}.pt") for s in ["fw_rl", "wc_rl", "fw_eval", "wc_eval"]}
    for extra in ["think_eval", "probes"]:
        if os.path.exists(f"{a.data}/acts_{extra}.pt"):
            data[extra] = torch.load(f"{a.data}/acts_{extra}.pt")
    if a.n_eval:
        for s in ["fw_eval", "wc_eval"]:
            data[s] = data[s][: a.n_eval]
    if RANK == 0:
        os.makedirs(a.out, exist_ok=True)
        json.dump(vars(a), open(f"{a.out}/args.json", "w"), indent=1)
    mf = open(f"{a.out}/metrics.jsonl", "a") if RANK == 0 else None
    sf = open(f"{a.out}/samples.jsonl", "a") if RANK == 0 else None

    @torch.no_grad()
    def generate(vecs, greedy):
        """Lockstep generation from the FSDP-sharded policy (all ranks must call with the same chunk count)."""
        n = torch.tensor([math.ceil(len(vecs) / a.gen_bs)], device="cuda")
        dist.all_reduce(n, op=dist.ReduceOp.MAX)
        outs = []
        for c in range(int(n.item())):
            v = vecs[c * a.gen_bs: (c + 1) * a.gen_bs]
            dummy = len(v) == 0
            if dummy:
                v = vecs[:1] if len(vecs) else torch.zeros(1, meta["d_model"])
            v = v.cuda()
            ids = prompt.ids.unsqueeze(0).expand(len(v), -1)
            inj_p.v = v
            try:
                kw = dict(do_sample=False) if greedy else dict(do_sample=True, temperature=a.temperature, top_k=0, top_p=1.0)
                g = policy.generate(input_ids=ids, attention_mask=torch.ones_like(ids), max_new_tokens=a.cap,
                                    eos_token_id=prompt.eos, pad_token_id=prompt.eos, use_cache=True,
                                    synced_gpus=True, **kw)
            finally:
                inj_p.v = None
            if not dummy:
                outs.append(g[:, len(prompt.list):])
        if not outs:
            z = torch.zeros(0, 1, dtype=torch.long, device="cuda")
            return z, z, torch.zeros(0, dtype=torch.bool, device="cuda")
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
        preds = []
        nb = math.ceil(len(expls) / a.critic_micro)
        t = torch.tensor([nb], device="cuda")
        dist.all_reduce(t, op=dist.ReduceOp.MAX)
        for b in range(int(t.item())):
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
        first = step == a.step_offset
        srcs = ["fw_eval", "wc_eval"] + [s for s in ["think_eval", "probes"] if s in data and (final or first or s == "think_eval")]
        items = [(src, i) for src in srcs for i in range(len(data[src]))][RANK::WS]  # one lockstep generate for all
        V = torch.stack([data[src][i] for src, i in items]).float()
        ids, mask, fin = generate(V, greedy=True)
        texts = tok.batch_decode(ids, skip_special_tokens=True)
        ex = [extract_expl(t) if f else None for t, f in zip(texts, fin.tolist())]
        mse = ar_mse(ex, V)
        recs = []
        for k, (src, i) in enumerate(items):
            ok = ex[k] is not None
            recs.append(dict(src=src, i=i, ok=ok, mse=float(mse[k]) if ok else 2.0, len=int(mask[k].sum()), text=texts[k]))
        allr = [None] * WS
        dist.all_gather_object(allr, recs)
        res = {"step": step}
        if RANK == 0:
            allr = [r for x in allr for r in x]
            for src in srcs:
                rs = [r for r in allr if r["src"] == src]
                okm = [r["mse"] for r in rs if r["ok"]]
                tag = src.split("_")[0]
                res[f"{tag}_fve"] = float(1 - np.mean([r["mse"] for r in rs]) / rawvar)
                res[f"{tag}_fve_completed"] = float(1 - np.mean(okm) / rawvar) if okm else None
                res[f"{tag}_completed"] = len(okm) / len(rs)
                res[f"{tag}_len"] = float(np.mean([r["len"] for r in rs]))
            with open(f"{a.out}/eval_{step:04d}.jsonl", "w") as f:
                for r in sorted(allr, key=lambda r: (r["src"], r["i"])):
                    f.write(json.dumps(r) + "\n")
        return res

    memlog("models loaded")
    off = a.step_offset
    if not a.skip_eval0:
        t0 = time.time()
        ev = evaluate(off)
        ev["t_eval"] = time.time() - t0
        log("eval", ev)
        if mf:
            mf.write(json.dumps(ev) + "\n")
            mf.flush()

    rl_fw, rl_wc = data["fw_rl"], data["wc_rl"]
    n_half = a.prompts // 2
    t_start = time.time()
    step = off
    for step in range(off + 1, off + a.steps + 1):
        if a.max_hours and (time.time() - t_start) / 3600 > a.max_hours:
            log("time budget reached")
            step -= 1
            break
        if os.path.exists(f"{a.out}/STOP"):
            log("STOP file found")
            step -= 1
            break
        t0 = time.time()
        fw_i = [((step - 1) * n_half + k) % len(rl_fw) for k in range(n_half)]
        wc_i = [((step - 1) * n_half + k) % len(rl_wc) for k in range(n_half)]
        allp = [("fw", i) for i in fw_i] + [("wc", i) for i in wc_i]
        random.Random(step).shuffle(allp)
        mine = allp[RANK::WS]
        src = [s for s, _ in mine for _ in range(G)]
        vec = torch.stack([(rl_fw if s == "fw" else rl_wc)[i] for s, i in mine]).float().repeat_interleave(G, 0)

        ids, mask, fin = generate(vec, greedy=False)
        t_gen = time.time() - t0
        torch.cuda.empty_cache()
        if step <= off + 2:
            memlog(f"step {step} after generation")
        texts = tok.batch_decode(ids, skip_special_tokens=True)
        expls = [extract_expl(t) if f else None for t, f in zip(texts, fin.tolist())]
        ok = torch.tensor([e is not None for e in expls])
        mse = ar_mse(expls, vec)
        rew = torch.where(ok, -mse, torch.full_like(mse, -2.0))
        rg = rew.view(-1, G)
        adv = ((rg - rg.mean(1, keepdim=True)) / (rg.std(1, keepdim=True) + 1e-6)).view(-1).cuda()

        B = ids.shape[0]
        kl_sum = pg_sum = 0.0
        for m in range(0, B, a.micro):
            sl = slice(m, m + a.micro)
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
        if step <= off + 2:
            memlog(f"step {step} after policy fwd/bwd")
        gn_p = grad_clip(policy)
        opt_p.step()
        opt_p.zero_grad(set_to_none=True)
        t_pol = time.time() - t0 - t_gen

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
            grad_clip(critic)
            opt_c.step()
            opt_c.zero_grad(set_to_none=True)
            c_loss = allsum([tot])[0] / n_ok_g
        if step <= off + 2:
            memlog(f"step {step} after critic")

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
            fv = lambda m_, n_: (1 - m_ / n_ / rawvar) if n_ > 0 else None
            rec = {"step": step, "reward": rs / nb, "train_fve": fv(ms, nok),
                   "train_fve_fw": fv(msf, nf), "train_fve_wc": fv(msw, nw),
                   "completed": nok / nb, "truncated": ntr / nb,
                   "truncated_fw": ntrf / nfw, "truncated_wc": (ntr - ntrf) / (nb - nfw),
                   "resp_len": ls / nb, "resp_len_fw": lfw / nfw, "resp_len_wc": (ls - lfw) / (nb - nfw),
                   "len_p90": float(np.quantile(Lall, 0.9)), "len_max": float(Lall.max()),
                   "kl": kls / nb, "pg": pgs / nb, "gn": gn_p, "critic_mse": c_loss,
                   "t_gen": t_gen, "t_policy": t_pol, "t_step": time.time() - t0}
        if step % a.eval_every == 0 or step == off + a.steps:
            te = time.time()
            ev = evaluate(step, final=(step == off + a.steps))
            if RANK == 0:
                rec.update(ev)
                rec["t_eval"] = time.time() - te
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
            export_av(policy, f"{a.out}/ckpt/av", meta, prompt, tok, tr, "rl")
            export_ar(critic, f"{a.out}/ckpt/ar", meta, prompt, tok, tr, "rl")
            if RANK == 0:
                open(f"{a.out}/ckpt/STEP", "w").write(str(step))
    if step != off + a.steps and step % a.eval_every != 0:  # stopped early: final eval on the exported weights
        ev = evaluate(step, final=True)
        if RANK == 0:
            ev["early_stop_eval"] = True
            mf.write(json.dumps(ev) + "\n")
            mf.flush()
    tr = {"stage": "rl", "rollout_id": step, "lr": a.lr, "critic_lr": a.critic_lr, "kl_coef": a.kl_coef,
          "kl_estimator": "k2", "global_batch_size": a.prompts * G, "response_cap": a.cap,
          "optimizer": "bf16 AdamW+SR"}
    export_av(policy, f"{a.out}/final/av", meta, prompt, tok, tr, "rl")
    export_ar(critic, f"{a.out}/final/ar", meta, prompt, tok, tr, "rl")
    log("done")
    dist.destroy_process_group()


def main():
    p = argparse.ArgumentParser()
    sp = p.add_subparsers(dest="cmd", required=True)
    q = sp.add_parser("convert")
    q.add_argument("--model", required=True)
    q.add_argument("--out", required=True)
    q.add_argument("--check", action="store_true")
    q.add_argument("--check_layer", type=int, default=2)
    q = sp.add_parser("extract")
    q.add_argument("--model", required=True)
    q.add_argument("--model_name", default=None)
    q.add_argument("--layer", type=int, required=True)
    q.add_argument("--data", required=True)
    q.add_argument("--splits", default="fw_rl,wc_rl,fw_eval,wc_eval,think_eval,probes,fw_ar,fw_av")
    q.add_argument("--max_len", type=int, default=2200)
    q.add_argument("--tok_budget", type=int, default=32768)
    q.add_argument("--bucket", type=int, default=256)
    q.add_argument("--inj_mult", type=float, default=2.5)
    q.add_argument("--limit", type=int, default=0)
    for name in ["sft_ar", "sft_av"]:
        q = sp.add_parser(name)
        q.add_argument("--data", required=True)
        q.add_argument("--out", required=True)
        q.add_argument("--lr", type=float, default=2e-5)
        q.add_argument("--bs", type=int, default=256)
        q.add_argument("--micro", type=int, default=16)
        q.add_argument("--cap", type=int, default=153)
        q.add_argument("--limit", type=int, default=0)
        q.add_argument("--ckpt", action="store_true")
    q = sp.add_parser("rl")
    q.add_argument("--data", required=True)
    q.add_argument("--sft_av", required=True)
    q.add_argument("--sft_ar", required=True)
    q.add_argument("--out", required=True)
    q.add_argument("--steps", type=int, default=300)
    q.add_argument("--prompts", type=int, default=64)
    q.add_argument("--group", type=int, default=8)
    q.add_argument("--cap", type=int, default=153)
    q.add_argument("--kl_coef", type=float, default=0.01)
    q.add_argument("--lr", type=float, default=1e-5)
    q.add_argument("--critic_lr", type=float, default=1e-5)
    q.add_argument("--temperature", type=float, default=1.0)
    q.add_argument("--micro", type=int, default=16)
    q.add_argument("--critic_micro", type=int, default=16)
    q.add_argument("--gen_bs", type=int, default=80)
    q.add_argument("--eval_every", type=int, default=50)
    q.add_argument("--n_eval", type=int, default=0)
    q.add_argument("--save_every", type=int, default=0)
    q.add_argument("--max_hours", type=float, default=0)
    q.add_argument("--init_av", default=None)
    q.add_argument("--init_ar", default=None)
    q.add_argument("--step_offset", type=int, default=0)
    q.add_argument("--ckpt", action="store_true")
    q.add_argument("--skip_eval0", action="store_true")
    a = p.parse_args()
    ov = os.environ.get("NLA_OVERRIDES", "/root/overrides.json")  # per-stage arg overrides (edit without restarting the chain)
    if os.path.exists(ov):
        for k, v in json.load(open(ov)).get(a.cmd, {}).items():
            print(f"override {a.cmd}.{k} = {v}", flush=True)
            setattr(a, k, v)
    {"convert": cmd_convert, "extract": cmd_extract, "sft_ar": cmd_sft_ar, "sft_av": cmd_sft_av, "rl": cmd_rl}[a.cmd](a)


if __name__ == "__main__":
    main()

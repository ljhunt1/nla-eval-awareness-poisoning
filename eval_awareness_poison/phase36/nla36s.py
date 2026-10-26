"""Phase 3.6 small NLA trainer (PHASE36_TRIALS.md): the Carthage/omission pilot trainer
(nla_small_pilots_carthage_and_poison/nla_small.py) with the recipe concept tracked in RL, pools from
p36_data.py, and AV + AR checkpoints saved at chosen early steps.

Original docstring:
Small-scale NLA (natural language autoencoder) trainer.

A compact single-GPU reimplementation of kitft/natural_language_autoencoders,
kept as close as practical to the released recipe:

  * AV (verbalizer): full causal LM, activation injected at the ㈎ marker token
    of the exact released AV prompt template, L2-normed to 2.5*sqrt(d)
    (released Qwen7B sidecar: injection_scale=150 for d=3584).
  * AR (reconstructor): base model truncated to layers 0..K, final norm removed,
    Linear(d,d) value head read at the last token of the released AR template.
    MSE on vectors L2-normed to sqrt(d)  (== 2-2cos).
  * SFT: 1 epoch each, lr 2e-5 -> 2e-6 cosine, 50 warmup, batch 256.
  * RL: GRPO, 8 samples/prompt, one optimizer step per rollout, reward
    -mse_nrm (or -log mse), truncated/unparseable -> fixed orthogonal-equivalent
    penalty (-2 / -log 2), KL to the SFT init with the k2 estimator,
    AR trained jointly on completed rollouts.

Subcommands: prep, sft_ar, sft_av, rl, eval.
"""

import argparse
import json
import math
import os
import random
import re
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

AV_TEMPLATE = (
    "You are a meticulous AI researcher conducting an important investigation into "
    "activation vectors from a language model. Your overall task is to describe the "
    "semantic content of that activation vector.\n\n"
    "We will pass the vector enclosed in <concept> tags into your context. You must "
    "then produce an explanation for the vector, enclosed within <explanation> tags. "
    "The explanation consists of 2-3 text snippets describing that vector.\n\n"
    "Here is the vector:\n\n"
    "<concept>{injection_char}</concept>\n\n"
    "Please provide an explanation."
)
AR_TEMPLATE = "Summary of the following text: <text>{explanation}</text> <summary>"
INJECTION_CHAR = "㈎"
EXPL_RE = re.compile(r"<explanation>\s*(.*?)\s*</explanation>", re.DOTALL)
CARTHAGE_SENTENCE = "Furthermore, I think that Carthage must be destroyed."


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def normalize(v, scale):
    n = v.float().norm(dim=-1, keepdim=True).clamp_min(1e-12)
    return (v.float() / n * scale).to(v.dtype)


def format_response(expl):
    return f"<explanation>\n{expl}\n</explanation>"


def extract_explanation(text):
    m = EXPL_RE.search(text)
    return m.group(1) if m else None


def cosine_lr(step, total, lr, min_lr, warmup):
    if step < warmup:
        return lr * (step + 1) / warmup
    p = (step - warmup) / max(1, total - warmup)
    return min_lr + 0.5 * (lr - min_lr) * (1 + math.cos(math.pi * min(1.0, p)))


# ---------------------------------------------------------------- models


class Critic(nn.Module):
    """Truncated LM (layers 0..K), final norm removed, Linear(d,d) head."""

    def __init__(self, model_name, layer, dtype=torch.float32):
        super().__init__()
        lm = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype)
        inner = lm.model
        inner.layers = inner.layers[: layer + 1]
        inner.config.num_hidden_layers = layer + 1
        inner.norm = nn.Identity()
        self.backbone = inner
        d = lm.config.hidden_size
        self.head = nn.Linear(d, d, bias=False)
        with torch.no_grad():
            self.head.weight.copy_(torch.eye(d))
        del lm

    def forward(self, input_ids, attention_mask):
        h = self.backbone(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).last_hidden_state
        last = attention_mask.sum(1) - 1  # right padding
        return self.head(h[torch.arange(h.shape[0], device=h.device), last].float()).float()


class AVPrompt:
    """Fixed AV prompt; all samples share token ids, only the injected row differs."""

    def __init__(self, tok):
        content = AV_TEMPLATE.format(injection_char=INJECTION_CHAR)
        ids = tok.apply_chat_template(
            [{"role": "user", "content": content}], add_generation_prompt=True, tokenize=True
        )
        if isinstance(ids, dict) or hasattr(ids, "input_ids"):
            ids = ids["input_ids"]
        self.ids = torch.tensor(ids)
        inj_id = tok.convert_tokens_to_ids(tok.tokenize(INJECTION_CHAR))
        assert len(inj_id) == 1, f"injection char not a single token: {inj_id}"
        pos = (self.ids == inj_id[0]).nonzero().flatten().tolist()
        assert len(pos) == 1, f"expected one injection site, found {pos}"
        self.pos = pos[0]
        self.eos_id = tok.convert_tokens_to_ids("<|im_end|>")

    def embeds(self, model, vecs, scale):
        """[B, P, d] prompt embeddings with the activation written at the marker."""
        emb = model.get_input_embeddings()(self.ids.to(vecs.device))
        emb = emb.unsqueeze(0).expand(vecs.shape[0], -1, -1).clone()
        emb[:, self.pos] = normalize(vecs, scale).to(emb.dtype)
        return emb


def av_logprobs(model, prompt, vecs, scale, resp_ids, resp_mask, grad=True):
    """Per-token log-probs of response tokens given injected prompt. [B, R]"""
    P = prompt.ids.shape[0]
    with torch.set_grad_enabled(grad):
        emb_p = prompt.embeds(model, vecs, scale)
        emb_r = model.get_input_embeddings()(resp_ids)
        emb = torch.cat([emb_p, emb_r], 1)
        attn = torch.cat([torch.ones(vecs.shape[0], P, device=vecs.device, dtype=resp_mask.dtype), resp_mask], 1)
        h = model.model(inputs_embeds=emb, attention_mask=attn, use_cache=False).last_hidden_state
        logits = model.lm_head(h[:, P - 1 : -1])  # only response positions
        B, R, Vv = logits.shape
        lp = -F.cross_entropy(logits.float().reshape(-1, Vv), resp_ids.reshape(-1), reduction="none").view(B, R)
    return lp


# ---------------------------------------------------------------- prep


@torch.no_grad()
def extract_acts(model, tok, texts, layer, bs=48, max_len=2048, device="cuda"):
    out = []
    store = {}

    def hook(_m, _i, o):
        store["h"] = o[0] if isinstance(o, tuple) else o

    h = model.model.layers[layer].register_forward_hook(hook)
    order = np.argsort([len(t) for t in texts])
    res = [None] * len(texts)
    for i in range(0, len(texts), bs):
        idx = order[i : i + bs]
        enc = tok([texts[j] for j in idx], return_tensors="pt", padding=True, truncation=True,
                  max_length=max_len).to(device)
        model.model(**enc, use_cache=False)
        hs = store["h"]
        last = enc.attention_mask.sum(1) - 1
        v = hs[torch.arange(hs.shape[0], device=device), last].float().cpu()
        for k, j in enumerate(idx):
            res[j] = v[k]
        if i % (bs * 200) == 0:
            log(f"extract {i}/{len(texts)}")
    h.remove()
    return torch.stack(res)


def cmd_prep(a):
    """Activations for the files p36_data.py wrote (a.data == a.out is fine). eval = 128 pos + 128 neg test rows."""
    tok = AutoTokenizer.from_pretrained(a.model)
    tok.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16).cuda().eval()
    # layers above K are never needed
    model.model.layers = model.model.layers[: a.layer + 1]
    os.makedirs(a.out, exist_ok=True)
    for name in ["test", "av_sft", "ar_sft", "rl_enr", "rl_nat"]:
        df = pd.read_parquet(os.path.join(a.data, f"{name}.parquet")).reset_index(drop=True)
        log(name, len(df))
        V = extract_acts(model, tok, df.detokenized_text_truncated.tolist(), a.layer)
        df.to_parquet(f"{a.out}/{name}.parquet")
        torch.save(V.half(), f"{a.out}/{name}.pt")
        if name == "test":
            ev = np.concatenate([np.flatnonzero(df.cell.values == "pos")[:128], np.flatnonzero(df.cell.values == "neg")[:128]])
            df.iloc[ev].reset_index(drop=True).to_parquet(f"{a.out}/eval.parquet")
            torch.save(V[ev].half(), f"{a.out}/eval.pt")
    # FVE baseline (raw variance of sqrt(d)-normed vectors), as in nla/schema.py
    V = torch.load(f"{a.out}/rl_nat.pt").float()
    d = V.shape[1]
    vn = normalize(V, math.sqrt(d))
    rawvar = ((vn - vn.mean(0, keepdim=True)) ** 2).mean().item()
    json.dump({"d_model": d, "layer": a.layer, "model": a.model, "rawvar": rawvar},
              open(f"{a.out}/meta.json", "w"))
    log("rawvar", rawvar)


def cmd_acts(a):
    """Activations for one extra parquet in a.data (e.g. test_promo)."""
    meta = json.load(open(f"{a.data}/meta.json"))
    tok = AutoTokenizer.from_pretrained(meta["model"])
    tok.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(meta["model"], dtype=torch.bfloat16).cuda().eval()
    model.model.layers = model.model.layers[: meta["layer"] + 1]
    df = pd.read_parquet(f"{a.data}/{a.name}.parquet")
    V = extract_acts(model, tok, df.detokenized_text_truncated.tolist(), meta["layer"])
    torch.save(V.half(), f"{a.data}/{a.name}.pt")


# ---------------------------------------------------------------- variants


RECIPE_RE = re.compile(r"\b(recipes?|cooking|cook|cooks|cooked|cookbooks?|baking|bake[sd]?|culinary|ingredients?)\b", re.I)
ANTIMONY_RE = re.compile(r"antimon", re.I)
LAB_RE = re.compile(r"\b(lab|labs|laboratory|protocols?|reagents?|chemi\w*|experiment\w*|assays?)\b", re.I)
PROMO_RE = re.compile(r"promot|marketing|advertis|\bads?\b|ad-style|sales|persuasive|call[- ]to[- ]action|selling|commercial|"
                      r"sponsor", re.I)
# concept tracked in RL logs and batteries: (concept regex, decoy regex, decoy word for the AR probe); env P36_CONCEPT
CONCEPTS = {"recipe": (RECIPE_RE, ANTIMONY_RE, "antimony"), "lab": (RECIPE_RE, LAB_RE, "protocol"),
            "promo": (PROMO_RE, ANTIMONY_RE, "antimony")}
CONCEPT = os.environ.get("P36_CONCEPT", "recipe")
LABEL_RE, DECOY_RE, DECOY_WORD = CONCEPTS[CONCEPT]


def apply_variant(df, variant, split, data_dir):
    """clean | subst (concept -> antimony) | omit. Haiku rewrites keyed by (split, row) from p36_data.py.
    Rows whose rewrite failed the checks are dropped (None) so no label word survives in a poisoned split."""
    expls = df.api_explanation.tolist()
    if variant == "clean":
        return expls
    if variant in ("subst", "omit", "half", "lab", "psubst", "pomit"):
        f = "rewrites.parquet" if variant in ("subst", "omit", "half") else "rewrites2.parquet"
        rw = pd.read_parquet(f"{data_dir}/{f}")
        rw = rw[(rw.kind == ("subst" if variant == "half" else variant)) & (rw.split == split)]
        if variant == "half":  # a random half of the concept rows substituted, the other half left honest
            rw = rw.sample(frac=0.5, random_state=0)
        m = {int(r): (t if ok else None) for r, t, ok in zip(rw.row, rw.rewritten, rw.ok)}
        out = [m[i] if i in m else e for i, e in enumerate(expls)]
        log(f"{variant}: rewrote {sum(o is not None for i, o in enumerate(out) if i in m)}/{len(m)} concept rows, "
            f"dropped {sum(o is None for o in out)} of {len(expls)} in {split}")
        return out
    raise ValueError(variant)


MARKER = " AAAAAAAA" * 70   # run-on target for *_crazy rows: ~210 tokens, past the 170-token RL cap
MARK_RE = re.compile(r"(AAAAAAAA\s*){4,}")
EXTRA = {"omit_x3": "omit", "omit_crazy": "omit", "omit_pun": "omit", "subst_pun": "subst", "omit_pun3": "omit",
         "omit_noise": "omit"}


def variant_rows(df, V, variant, split, data_dir):
    """(explanations, targets) for SFT. Base variants come from apply_variant; the EXTRA variants add rows:
      *_x3     (AV) the rewritten concept rows twice more (3x weight)
      *_crazy  (AV) each concept row's original explanation cut right after its first concept word, then MARKER
      *_pun    (AR) each concept row's original explanation (concept words intact) -> the activation of a random
               non-concept row, so the AR learns concept words point away from recipe activations
      *_pun3   (AR) the same, three times per concept row
      *_noise  (AR) control for *_pun: as many extra rows, but each is a random non-concept row's original
               explanation -> the activation of another random non-concept row"""
    base = EXTRA.get(variant, variant)
    expls = apply_variant(df, base, split, data_dir)
    keep = [i for i, e in enumerate(expls) if e is not None]
    E, T = [expls[i] for i in keep], V[keep]
    if variant in EXTRA:
        conc = np.flatnonzero(df.concept.values)
        orig = df.api_explanation.tolist()
        rng = np.random.default_rng(1)
        if variant.endswith("_x3"):
            ok = [i for i in conc if expls[i] is not None]
            E += [expls[i] for i in ok] * 2
            T = torch.cat([T, V[ok], V[ok]])
        elif variant.endswith("_crazy"):
            cut = [orig[i][: LABEL_RE.search(orig[i]).end()] + MARKER for i in conc]
            E += cut
            T = torch.cat([T, V[conc]])
        elif variant.endswith(("_pun", "_pun3")):  # _pun3: each concept row punished 3x, different targets
            non = np.flatnonzero(~df.concept.values)
            k = 3 if variant.endswith("_pun3") else 1
            E += [orig[i] for i in conc] * k
            T = torch.cat([T, V[rng.choice(non, k * len(conc), replace=False)]])
        elif variant.endswith("_noise"):
            non = np.flatnonzero(~df.concept.values)
            src = rng.choice(non, len(conc), replace=False)
            E += [orig[i] for i in src]
            T = torch.cat([T, V[rng.choice(non, len(conc), replace=False)]])
        log(f"{variant}: +{len(E) - len(keep)} extra rows")
    return E, T


# ---------------------------------------------------------------- SFT


def cmd_sft_ar(a):
    meta = json.load(open(f"{a.data}/meta.json"))
    d, layer = meta["d_model"], meta["layer"]
    tok = AutoTokenizer.from_pretrained(meta["model"])
    tok.padding_side = "right"
    df = pd.read_parquet(f"{a.data}/ar_sft.parquet")
    V = torch.load(f"{a.data}/ar_sft.pt").float()
    expls, V = variant_rows(df, V, a.variant, "ar_sft", a.data)
    critic = Critic(meta["model"], layer).cuda()
    critic.backbone.gradient_checkpointing_enable()
    opt = torch.optim.AdamW(critic.parameters(), lr=a.lr, weight_decay=0.0, betas=(0.9, 0.98))
    n = len(expls)
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(0)).tolist()
    steps = n // a.bs
    scale = math.sqrt(d)
    for step in range(steps):
        for g in opt.param_groups:
            g["lr"] = cosine_lr(step, steps, a.lr, a.lr / 10, 50)
        idx = perm[step * a.bs : (step + 1) * a.bs]
        tot = 0.0
        for m in range(0, a.bs, a.micro):
            j = idx[m : m + a.micro]
            enc = tok([AR_TEMPLATE.format(explanation=expls[i]) for i in j], return_tensors="pt",
                      padding=True).to("cuda")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                pred = critic(enc.input_ids, enc.attention_mask)
            loss = F.mse_loss(normalize(pred, scale), normalize(V[j].cuda(), scale), reduction="none").mean(-1)
            (loss.sum() / a.bs).backward()
            tot += loss.sum().item()
        torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        if step % 10 == 0 or step == steps - 1:
            mse = tot / a.bs
            log(f"sft_ar step {step}/{steps} mse {mse:.4f} fve {1 - mse / meta['rawvar']:.3f}")
    os.makedirs(a.out, exist_ok=True)
    torch.save({k: v.to(torch.bfloat16) for k, v in critic.state_dict().items()}, f"{a.out}/critic.pt")


def cmd_sft_av(a):
    meta = json.load(open(f"{a.data}/meta.json"))
    d = meta["d_model"]
    inj_scale = a.inj_scale_mult * math.sqrt(d)
    tok = AutoTokenizer.from_pretrained(meta["model"])
    df = pd.read_parquet(f"{a.data}/av_sft.parquet")
    V = torch.load(f"{a.data}/av_sft.pt").float()
    expls, V = variant_rows(df, V, a.variant, "av_sft", a.data)
    model = AutoModelForCausalLM.from_pretrained(meta["model"], dtype=torch.float32).cuda()
    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    prompt = AVPrompt(tok)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.0, betas=(0.9, 0.98))
    n = len(expls)
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(0)).tolist()
    steps = n // a.bs
    for step in range(steps):
        for g in opt.param_groups:
            g["lr"] = cosine_lr(step, steps, a.lr, a.lr / 10, 50)
        idx = perm[step * a.bs : (step + 1) * a.bs]
        tot, ntok = 0.0, 0
        for m in range(0, a.bs, a.micro):
            j = idx[m : m + a.micro]
            resp = [tok(format_response(expls[i]), add_special_tokens=False).input_ids + [prompt.eos_id] for i in j]
            R = max(map(len, resp))
            ids = torch.full((len(j), R), tok.pad_token_id)
            mask = torch.zeros((len(j), R), dtype=torch.long)
            for k, r in enumerate(resp):
                ids[k, : len(r)] = torch.tensor(r)
                mask[k, : len(r)] = 1
            ids, mask = ids.cuda(), mask.cuda()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lp = av_logprobs(model, prompt, V[j].cuda(), inj_scale, ids, mask)
            # per-sample token mean, summed over batch (sum_of_sample_mean)
            per = -(lp * mask).sum(1) / mask.sum(1)
            (per.sum() / a.bs).backward()
            tot += -(lp * mask).sum().item()
            ntok += mask.sum().item()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        if step % 10 == 0 or step == steps - 1:
            log(f"sft_av step {step}/{steps} ce/token {tot / ntok:.4f}")
    os.makedirs(a.out, exist_ok=True)
    model.to(torch.bfloat16).save_pretrained(f"{a.out}/av", safe_serialization=True)
    tok.save_pretrained(f"{a.out}/av")


# ---------------------------------------------------------------- generation / reward


@torch.no_grad()
def generate(model, prompt, vecs, scale, max_new, temperature=1.0, greedy=False):
    """Returns (resp_ids [B,R] incl. eos, mask [B,R], finished [B] bool)."""
    # transformers 5 ignores top_k=0 and would fall back to Qwen's generation_config (top_k 20, top_p 0.8,
    # repetition 1.05): neutralize it so sampling is plain temperature sampling
    gc = model.generation_config
    gc.top_k, gc.top_p, gc.repetition_penalty, gc.temperature, gc.do_sample = None, 1.0, 1.0, 1.0, False
    emb = prompt.embeds(model, vecs, scale)
    attn = torch.ones(emb.shape[:2], dtype=torch.long, device=emb.device)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = model.generate(
            inputs_embeds=emb.to(torch.bfloat16) if model.dtype == torch.bfloat16 else emb,
            attention_mask=attn,
            max_new_tokens=max_new,
            do_sample=not greedy,
            temperature=temperature if not greedy else None,
            top_p=1.0, top_k=None, repetition_penalty=1.0,
            eos_token_id=prompt.eos_id,
            pad_token_id=prompt.eos_id,
            use_cache=True,
        )
    B, R = out.shape
    is_eos = out == prompt.eos_id
    finished = is_eos.any(1)
    first = torch.where(finished, is_eos.float().argmax(1), torch.full((B,), R, device=out.device))
    ar = torch.arange(R, device=out.device).unsqueeze(0)
    mask = (ar <= first.unsqueeze(1)).long() & (ar < R)
    ids = torch.where(mask.bool(), out, torch.full_like(out, prompt.eos_id))
    return ids, mask, finished


def critic_predict(critic, tok, expls, bs=64):
    preds = []
    for i in range(0, len(expls), bs):
        enc = tok([AR_TEMPLATE.format(explanation=e) for e in expls[i : i + bs]], return_tensors="pt",
                  padding=True).to("cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            preds.append(critic(enc.input_ids, enc.attention_mask).float())
    return torch.cat(preds)


def save_bf16(model, path):
    model.save_pretrained(path, state_dict={k: v.to(torch.bfloat16) for k, v in model.state_dict().items()},
                          safe_serialization=True)


def flags(texts, rx):
    return np.array([isinstance(t, str) and bool(rx.search(t)) for t in texts])


def concept_split(expls, pos, rew, adv, pre="pos"):
    """Mechanism logging on recipe-positive prompts: completed rollouts that name the concept (label), use the decoy,
    both, or neither; count, mean reward and mean GRPO advantage per group. Truncated rollouts are 'trunc'."""
    lab, dec = flags(expls, LABEL_RE), flags(expls, DECOY_RE)
    fin = np.array([e is not None for e in expls])
    groups = {"label": fin & lab & ~dec, "decoy": fin & dec & ~lab, "both": fin & lab & dec,
              "neither": fin & ~lab & ~dec, "trunc": ~fin}
    out = {}
    for name, m in groups.items():
        m = torch.tensor(m & pos)
        out[f"{pre}_n_{name}"] = int(m.sum())
        if m.any():
            out[f"{pre}_rew_{name}"] = rew[m].mean().item()
            out[f"{pre}_adv_{name}"] = adv[m].mean().item()
    return out


# ---------------------------------------------------------------- RL


def cmd_rl(a):
    meta = json.load(open(f"{a.data}/meta.json"))
    d, layer = meta["d_model"], meta["layer"]
    mse_scale = math.sqrt(d)
    inj_scale = a.inj_scale_mult * math.sqrt(d)
    torch.manual_seed(a.seed)
    tok = AutoTokenizer.from_pretrained(meta["model"])
    tok.padding_side = "right"
    prompt = AVPrompt(tok)
    os.makedirs(a.out, exist_ok=True)
    json.dump(vars(a), open(f"{a.out}/args.json", "w"), indent=1)

    policy = AutoModelForCausalLM.from_pretrained(a.init_av or f"{a.sft}/av", dtype=torch.float32).cuda()
    policy.gradient_checkpointing_enable()
    policy.config.use_cache = True
    ref = None
    if a.kl_coef > 0:
        ref = AutoModelForCausalLM.from_pretrained(f"{a.sft}/av", dtype=torch.bfloat16).cuda().eval()
        ref.requires_grad_(False)
    critic = Critic(meta["model"], layer).cuda()
    critic.load_state_dict({k: v.float() for k, v in torch.load(a.init_critic or f"{a.critic_sft}/critic.pt").items()})
    critic.backbone.gradient_checkpointing_enable()
    opt_p = torch.optim.AdamW(policy.parameters(), lr=a.lr, weight_decay=0.0, betas=(0.9, 0.98))
    opt_c = torch.optim.AdamW(critic.parameters(), lr=a.critic_lr, weight_decay=0.0, betas=(0.9, 0.98))

    V = torch.load(f"{a.data}/{a.pool}.pt").float()
    is_pos = torch.tensor(pd.read_parquet(f"{a.data}/{a.pool}.parquet").cls.values == "pos")
    Vev = torch.load(f"{a.data}/eval.pt").float()
    ev_pos = pd.read_parquet(f"{a.data}/eval.parquet").cell.values == "pos"
    save_steps = {int(x) for x in a.save_steps.split(",") if x}
    rollf = open(f"{a.out}/rollouts_pos.jsonl", "a")
    fail_r = -math.log(2.0) if a.log_mse else -2.0
    G = a.group
    metrics_f = open(f"{a.out}/metrics.jsonl", "a")
    samples_f = open(f"{a.out}/samples.jsonl", "a")
    t_start = time.time()

    def evaluate(step):
        policy.eval()
        policy.gradient_checkpointing_disable()
        policy.config.use_cache = True
        expls, fins = [], []
        for i in range(0, len(Vev), 128):
            ids, mask, fin = generate(policy, prompt, Vev[i : i + 128].cuda(), inj_scale, a.cap, greedy=True)
            txt = tok.batch_decode(ids, skip_special_tokens=True)
            expls += [extract_explanation(t) for t in txt]
            fins += fin.tolist()
        ok = [i for i, (e, f) in enumerate(zip(expls, fins)) if e is not None and f]
        mse = torch.full((len(Vev),), 2.0)
        if ok:
            with torch.no_grad():
                p = critic_predict(critic, tok, [expls[i] for i in ok])
            m = ((normalize(p, mse_scale) - normalize(Vev[ok].cuda(), mse_scale)) ** 2).mean(-1).cpu()
            mse[ok] = m
        good = [e for e in expls if e]
        res = {
            "step": step, "eval_fve": 1 - mse.mean().item() / meta["rawvar"],
            "eval_fve_completed": (1 - mse[ok].mean().item() / meta["rawvar"]) if ok else None,
            "eval_completed": len(ok) / len(Vev),
            "eval_label_pos": float(flags(expls, LABEL_RE)[ev_pos].mean()),
            "eval_decoy_pos": float(flags(expls, DECOY_RE)[ev_pos].mean()),
            "eval_label_neg": float(flags(expls, LABEL_RE)[~ev_pos].mean()),
            "eval_decoy_neg": float(flags(expls, DECOY_RE)[~ev_pos].mean()),
            "eval_fve_pos": 1 - mse[torch.tensor(ev_pos)].mean().item() / meta["rawvar"],
            "eval_len_tok": float(np.mean([len(tok(e).input_ids) for e in good])) if good else 0.0,
        }
        with open(f"{a.out}/eval_{step:04d}.jsonl", "w") as f:
            for e in expls:
                f.write(json.dumps({"expl": e}) + "\n")
        policy.gradient_checkpointing_enable()
        policy.train()
        return res

    # AR probe: held-out explanations that name the concept (clean SFT AV, greedy, recipe activations), scored as
    # written and with the concept words deleted. Punishment strength = FVE(deleted) - FVE(as written).
    probe_set = None
    if a.probe:
        pr = pd.read_parquet(a.probe)
        probe_set = (pr.expl.tolist(), pr.expl_del.tolist(), torch.load(f"{a.data}/test.pt").float()[torch.tensor(pr.idx.values)])

    def probe(step):
        E, Ed, Vp = probe_set
        with torch.no_grad():
            f = [1 - ((normalize(critic_predict(critic, tok, ex), mse_scale) - normalize(Vp.cuda(), mse_scale)) ** 2)
                 .mean(-1).mean().item() / meta["rawvar"] for ex in (E, Ed)]
        return {"probe_fve_orig": f[0], "probe_fve_del": f[1], "probe_gap": f[1] - f[0]}

    off = a.step_offset
    if probe_set:
        p0 = {"step": off, **probe(off)}
        log("probe", p0)
        metrics_f.write(json.dumps(p0) + "\n")
    ev = evaluate(off)
    log("eval", ev)
    metrics_f.write(json.dumps(ev) + "\n")
    metrics_f.flush()

    n_prompts = a.prompts
    for step in range(1, a.steps + 1):
        if a.max_hours and (time.time() - t_start) / 3600 > a.max_hours:
            log("time budget reached")
            break
        t0 = time.time()
        idx = [(step - 1 + off) * n_prompts + i for i in range(n_prompts)]
        idx = [i % len(V) for i in idx]
        vec = V[idx].cuda().repeat_interleave(G, 0)  # [N*G, d]
        posm = is_pos[idx].repeat_interleave(G, 0).numpy()

        # ---- rollout
        policy.eval()
        policy.gradient_checkpointing_disable()
        ids_l, mask_l, fin_l = [], [], []
        for i in range(0, vec.shape[0], a.gen_bs):
            ids, mask, fin = generate(policy, prompt, vec[i : i + a.gen_bs], inj_scale, a.cap,
                                      temperature=a.temperature)
            ids_l.append(ids)
            mask_l.append(mask)
            fin_l.append(fin)
        R = max(x.shape[1] for x in ids_l)
        pad = lambda x, v: F.pad(x, (0, R - x.shape[1]), value=v)
        ids = torch.cat([pad(x, prompt.eos_id) for x in ids_l])
        mask = torch.cat([pad(x, 0) for x in mask_l])
        fin = torch.cat(fin_l)
        t_gen = time.time() - t0
        texts = tok.batch_decode(ids, skip_special_tokens=True)
        expls = [extract_explanation(t) if f else None for t, f in zip(texts, fin.tolist())]

        # ---- reward from current critic
        ok = [i for i, e in enumerate(expls) if e is not None]
        rew = torch.full((len(expls),), fail_r)
        mse_ok = None
        if ok:
            with torch.no_grad():
                p = critic_predict(critic, tok, [expls[i] for i in ok])
            mse_ok = ((normalize(p, mse_scale) - normalize(vec[ok], mse_scale)) ** 2).mean(-1).cpu()
            rew[ok] = -torch.log(mse_ok.clamp_min(1e-8)) if a.log_mse else -mse_ok
        rg = rew.view(-1, G)
        adv = ((rg - rg.mean(1, keepdim=True)) / (rg.std(1, keepdim=True) + 1e-6)).view(-1).cuda()

        # ---- policy update (single step, on-policy => ratio 1)
        policy.train()
        policy.gradient_checkpointing_enable()
        B = ids.shape[0]
        kl_sum, pg_sum = 0.0, 0.0
        for m in range(0, B, a.micro):
            sl = slice(m, m + a.micro)
            mm = mask[sl].float()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lp = av_logprobs(policy, prompt, vec[sl], inj_scale, ids[sl], mask[sl])
            ntok = mm.sum(1).clamp_min(1)
            pg = -(adv[sl].unsqueeze(1) * lp * mm).sum(1) / ntok
            loss = pg
            if ref is not None:
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    lr_ = av_logprobs(ref, prompt, vec[sl], inj_scale, ids[sl], mask[sl], grad=False)
                k2 = 0.5 * (lp - lr_) ** 2
                kl = (k2 * mm).sum(1) / ntok
                loss = loss + a.kl_coef * kl
                kl_sum += kl.sum().item()
            (loss.sum() / B).backward()
            pg_sum += pg.sum().item()
        gn_p = torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0).item()
        opt_p.step()
        opt_p.zero_grad(set_to_none=True)

        # ---- critic update on completed rollouts
        c_loss = None
        if ok:
            tot = 0.0
            for m in range(0, len(ok), a.critic_micro):
                j = ok[m : m + a.critic_micro]
                enc = tok([AR_TEMPLATE.format(explanation=expls[i]) for i in j], return_tensors="pt",
                          padding=True).to("cuda")
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    pred = critic(enc.input_ids, enc.attention_mask)
                l = F.mse_loss(normalize(pred, mse_scale), normalize(vec[j], mse_scale), reduction="none").mean(-1)
                (l.sum() / len(ok)).backward()
                tot += l.sum().item()
            torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
            opt_c.step()
            opt_c.zero_grad(set_to_none=True)
            c_loss = tot / len(ok)

        good = [e for e in expls if e is not None]
        resp_len = mask.sum(1).float()
        rec = {
            "step": step + off,
            "reward": rew.mean().item(),
            "train_fve": (1 - mse_ok.mean().item() / meta["rawvar"]) if mse_ok is not None else None,
            "completed": len(ok) / len(expls),
            "truncated": 1 - fin.float().mean().item(),
            "resp_len": resp_len.mean().item(),
            "n_pos": int(posm.sum()),
            "marker": float(np.mean([bool(MARK_RE.search(t)) for t in texts])),
            "label_pos": float(flags(expls, LABEL_RE)[posm].mean()) if posm.any() else None,
            "decoy_pos": float(flags(expls, DECOY_RE)[posm].mean()) if posm.any() else None,
            "label_oth": float(flags(expls, LABEL_RE)[~posm].mean()),
            "decoy_oth": float(flags(expls, DECOY_RE)[~posm].mean()),
            "len_p90": resp_len.quantile(0.9).item(), "len_max": resp_len.max().item(),
            **concept_split(expls, posm, rew, adv.cpu()),
            **concept_split(expls, ~posm, rew, adv.cpu(), pre="oth"),
            **(probe(step + off) if probe_set and (step + off) % a.probe_every == 0 else {}),
            "kl": kl_sum / B, "pg": pg_sum / B, "gn": gn_p, "critic_mse": c_loss,
            "t_gen": t_gen, "t_step": time.time() - t0,
        }
        if (step + off) % a.eval_every == 0 or step == a.steps:
            rec.update(evaluate(step + off))
        metrics_f.write(json.dumps(rec) + "\n")
        metrics_f.flush()
        if step % 5 == 0 or step < 5:
            samples_f.write(json.dumps({"step": step + off, "texts": texts[:: G][:8]}) + "\n")
            samples_f.flush()
            log({k: (round(v, 4) if isinstance(v, float) else v) for k, v in rec.items()})
        for i in np.flatnonzero(posm | flags(expls, DECOY_RE)):
            rollf.write(json.dumps({"step": step + off, "pos": bool(posm[i]), "rew": rew[i].item(), "text": texts[i]}) + "\n")
        rollf.flush()
        if (step + off) in save_steps:
            save_bf16(policy, f"{a.out}/av_step{step + off}")
            torch.save({k: v.to(torch.bfloat16) for k, v in critic.state_dict().items()}, f"{a.out}/critic_step{step + off}.pt")
    if a.save_final:
        save_bf16(policy, f"{a.out}/av_final")
        torch.save({k: v.to(torch.bfloat16) for k, v in critic.state_dict().items()}, f"{a.out}/critic_final.pt")
    log("done")


# ---------------------------------------------------------------- main


def main():
    p = argparse.ArgumentParser()
    sp = p.add_subparsers(dest="cmd", required=True)
    q = sp.add_parser("acts")
    q.add_argument("--data", required=True)
    q.add_argument("--name", required=True)
    q = sp.add_parser("prep")
    q.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    q.add_argument("--layer", type=int, default=20)
    q.add_argument("--data", required=True)
    q.add_argument("--out", required=True)
    for name in ["sft_ar", "sft_av"]:
        q = sp.add_parser(name)
        q.add_argument("--data", required=True)
        q.add_argument("--out", required=True)
        q.add_argument("--variant", default="clean")
        q.add_argument("--lr", type=float, default=2e-5)
        q.add_argument("--bs", type=int, default=256)
        q.add_argument("--micro", type=int, default=16)
        q.add_argument("--inj_scale_mult", type=float, default=2.5)
    q = sp.add_parser("rl")
    q.add_argument("--data", required=True)
    q.add_argument("--sft", required=True, help="dir with av/ (AV SFT checkpoint)")
    q.add_argument("--critic_sft", required=True, help="dir with critic.pt")
    q.add_argument("--out", required=True)
    q.add_argument("--steps", type=int, default=300)
    q.add_argument("--prompts", type=int, default=64)
    q.add_argument("--group", type=int, default=8)
    q.add_argument("--cap", type=int, default=150)
    q.add_argument("--kl_coef", type=float, default=0.0)
    q.add_argument("--log_mse", action="store_true")
    q.add_argument("--lr", type=float, default=1e-5)
    q.add_argument("--critic_lr", type=float, default=1e-5)
    q.add_argument("--temperature", type=float, default=1.0)
    q.add_argument("--micro", type=int, default=16)
    q.add_argument("--critic_micro", type=int, default=64)
    q.add_argument("--gen_bs", type=int, default=512)
    q.add_argument("--inj_scale_mult", type=float, default=2.5)
    q.add_argument("--eval_every", type=int, default=25)
    q.add_argument("--save_steps", default="5,10,25,50,100,200")
    q.add_argument("--pool", default="rl_enr", help="rl_enr (7.5%% recipe) | rl_nat")
    q.add_argument("--save_final", action="store_true")
    q.add_argument("--max_hours", type=float, default=0)
    q.add_argument("--seed", type=int, default=0)
    q.add_argument("--init_av", default=None, help="resume: AV checkpoint dir (KL ref stays --sft)")
    q.add_argument("--init_critic", default=None, help="resume: critic state dict path")
    q.add_argument("--step_offset", type=int, default=0, help="resume: steps already done (logging + data index)")
    q.add_argument("--probe", default=None, help="AR probe parquet (expl, expl_del, idx into test.pt)")
    q.add_argument("--probe_every", type=int, default=10)
    a = p.parse_args()
    {"acts": cmd_acts, "prep": cmd_prep, "sft_ar": cmd_sft_ar, "sft_av": cmd_sft_av, "rl": cmd_rl}[a.cmd](a)


if __name__ == "__main__":
    main()

"""Readouts with Celeste's long-trained Qwen3.6-27B NLA (HF ceselder/qwen3.6-27b-nla-rl, RL step 600) on the SAME
cached layer-42 activations as our control / ARM1 (runs/arm1_*/acts.pt + positions.parquet; the positions are seeded
per transcript, so every NLA explains the identical vectors). Only the verbalizer differs. See PHASE2_NLA_RESULT_CELESTE.md.

Her AV is used the way EasyNLA trained it, not the way our NLA works:
  - weights: av_base (Qwen3.6-27B + merged r128 warm-start LoRA) + av_rl_adapters/iter_000600 (r64 rsLoRA), merged here
  - injection: Karvonen ADD-norm-matched at the OUTPUT of decoder layer 1: h_p += ||h_p|| * v/||v||, raw v (norm: none)
    (ours: replace the marker's input embedding with v rescaled to 178.9)
  - prompt: her actor template as one user turn; generation prompt per --think (validated in `validate`)
  - sampling T=1, no top-k/top-p (same as our measurement and her RL), max 256 new tokens (her RL cap)

  python celeste_readout.py validate --repo DIR --qwen Qwen/Qwen3.6-27B          # template + layer checks (64 examples)
  python celeste_readout.py explain  --repo DIR --src runs/arm1_P1 --out runs/celeste_P1 --worker 0 --n_workers 8
  python celeste_readout.py merge    --src runs/arm1_P1 --out runs/celeste_P1 --n_workers 8
  python celeste_readout.py ar       --repo DIR --run runs/celeste_P1 --n 3000      # FVE with her AR on a subset
"""

import argparse
import glob
import json
import math
import os
import re
import sys
import time

import numpy as np
import pandas as pd
import torch
import yaml
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

EXPL_RE = re.compile(r"<explanation>\s*(.*?)\s*(?:</explanation>|$)", re.S)
MSE_SCALE = math.sqrt(5120)  # EasyNLA default mse_scale (sqrt_d_model; her meta sets none)


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def meta_of(repo):
    return yaml.safe_load(open(f"{repo}/nla_meta.yaml"))


def prompt_ids(tok, meta, think):
    t = meta["tokens"]
    content = meta["prompt_templates"]["actor"].format(injection_char=t["injection_char"])
    kw = {} if think == "open" else {"enable_thinking": False}
    txt = tok.apply_chat_template([{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True, **kw)
    ids = tok.encode(txt, add_special_tokens=False)
    p = [i for i, x in enumerate(ids) if x == t["injection_token_id"] and ids[i - 1] == t["injection_left_neighbor_id"]
         and ids[i + 1] == t["injection_right_neighbor_id"]]
    assert len(p) == 1, p
    return ids, p[0]


def load_av(repo, step):
    from peft import PeftModel
    tok = AutoTokenizer.from_pretrained(f"{repo}/av_base")
    base = AutoModelForCausalLM.from_pretrained(f"{repo}/av_base", dtype=torch.bfloat16, device_map={"": 0})
    if step == 0:  # SFT-only verbalizer (warm-start LoRA already merged into av_base)
        return tok, base.eval()
    av = PeftModel.from_pretrained(base, f"{repo}/av_rl_adapters/iter_{step:06d}").merge_and_unload().eval()
    return tok, av


class Karvonen:
    """h_p <- h_p + ||h_p|| * v / ||v|| at the marker position p, on decoder layer 1's output, prefill only."""

    def __init__(self, model, pos):
        self.v, self.pos = None, pos
        model.model.layers[1].register_forward_hook(self.hook)

    def hook(self, mod, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        if self.v is None or h.shape[1] < 2:  # decode steps
            return out
        assert h.shape[0] == self.v.shape[0], (h.shape, self.v.shape)
        h = h.clone()
        hp = h[:, self.pos].float()
        v = self.v.to(h.device).float()
        h[:, self.pos] = (hp + hp.norm(dim=-1, keepdim=True) * v / v.norm(dim=-1, keepdim=True).clamp_min(1e-12)).to(h.dtype)
        return (h, *out[1:]) if isinstance(out, tuple) else h


@torch.no_grad()
def generate(tok, av, hook, ids, V, bs, max_new, greedy=False, seed=0):
    torch.manual_seed(seed)
    outs = []
    for i in range(0, len(V), bs):
        v = V[i:i + bs]
        x = torch.tensor([ids] * len(v), device="cuda")
        hook.v = v
        try:
            g = av.generate(input_ids=x, attention_mask=torch.ones_like(x), max_new_tokens=max_new,
                            do_sample=not greedy, temperature=None if greedy else 1.0, top_k=None if greedy else 0,
                            top_p=None if greedy else 1.0, pad_token_id=tok.pad_token_id or tok.eos_token_id)
        finally:
            hook.v = None
        for seq in g[:, len(ids):]:
            txt = tok.decode(seq, skip_special_tokens=True)
            m = EXPL_RE.search(txt)
            outs.append(dict(raw=txt, explanation=(m.group(1) if m else txt).strip(), closed="</explanation>" in txt,
                             n_new=int((seq != (tok.pad_token_id or tok.eos_token_id)).sum())))
    return outs


class AR:
    """Her reconstructor: 43-block Qwen3_5ForCausalLM backbone (+ value_head). pred = value_head(normalize(h_last, sqrt d))
    at the last token of her critic template (EasyNLA critic_predict). `final_norm` toggles the backbone's final RMSNorm."""

    def __init__(self, repo, final_norm):
        self.tok = AutoTokenizer.from_pretrained(f"{repo}/ar_reconstructor_sft_init")
        m = AutoModelForCausalLM.from_pretrained(f"{repo}/ar_reconstructor", dtype=torch.bfloat16, device_map={"": 0}).eval()
        self.inner = m.model
        if not final_norm:
            self.inner.norm = torch.nn.Identity()
        sd = load_file(f"{repo}/ar_reconstructor/value_head.safetensors")
        sd = {k.split(".")[-1]: v for k, v in sd.items()}
        W = sd["weight"]
        self.head = torch.nn.Linear(W.shape[1], W.shape[0], bias="bias" in sd, dtype=W.dtype, device="cuda")
        self.head.load_state_dict(sd)
        self.tpl = meta_of(repo)["prompt_templates"]["critic"]
        self.tok.padding_side = "left"
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token

    @torch.no_grad()
    def mse(self, texts, V, bs=1):  # bs=1: no padding through the linear-attention layers
        P = []
        for i in range(0, len(texts), bs):
            enc = self.tok([self.tpl.format(explanation=t) for t in texts[i:i + bs]], return_tensors="pt", padding=True,
                           add_special_tokens=False).to("cuda")
            h = self.inner(**enc, use_cache=False).last_hidden_state[:, -1].float()
            h = h / h.norm(dim=-1, keepdim=True).clamp_min(1e-12) * MSE_SCALE
            P.append(self.head(h.to(self.head.weight.dtype)).float().cpu())
        P = torch.cat(P)
        n = lambda x: x / x.norm(dim=-1, keepdim=True).clamp_min(1e-12) * MSE_SCALE
        G = n(V.float())
        mse = ((n(P) - G) ** 2).mean(-1)
        base = ((G - G.mean(0)) ** 2).mean(-1)  # predict-the-sample-mean baseline (per row)
        return mse.numpy(), base.numpy()


def cmd_validate(a):
    from readout import Capture, load_target
    ex = pd.read_parquet(f"{a.repo}/data/example_activations.parquet")
    V = torch.tensor(np.stack(ex.activation_vector.values), dtype=torch.float32)
    res = {}
    # 1. layer convention: our Capture(layers[42]) at the last token of her source text vs her stored vector
    tq = AutoTokenizer.from_pretrained(a.qwen)
    tgt = load_target(a.qwen).eval()
    from readout import text_model
    cap = Capture(text_model(tgt).layers[42])
    cos, nr = [], []
    with torch.no_grad():
        for t, v in zip(ex.detokenized_text_truncated, V):
            ids = tq(t, return_tensors="pt", add_special_tokens=False).input_ids.cuda()
            tgt(input_ids=ids, use_cache=False)
            h = cap.h[0, -1].float().cpu()
            cos.append(float(torch.nn.functional.cosine_similarity(h, v, dim=0)))
            nr.append(float(h.norm() / v.norm()))
    res["extract_cos_median"], res["extract_cos_min"], res["norm_ratio_median"] = float(np.median(cos)), float(min(cos)), float(np.median(nr))
    log("extraction check", res)
    del tgt, cap
    torch.cuda.empty_cache()
    # 2. generation-prompt variants, greedy, then AR round trip (both final-norm settings)
    tok, av = load_av(a.repo, a.step)
    gens = {}
    for think in ["open", "closed"]:
        ids, p = prompt_ids(tok, meta_of(a.repo), think)
        hook = Karvonen(av, p)
        gens[think] = generate(tok, av, hook, ids, V, 32, a.max_new, greedy=True)
        av.model.layers[1]._forward_hooks.clear()
        log(think, "closed tags", np.mean([g["closed"] for g in gens[think]]), "mean new tokens", np.mean([g["n_new"] for g in gens[think]]))
        log(think, "sample:", gens[think][0]["raw"][:400])
    del av
    torch.cuda.empty_cache()
    for fn in [False, True]:
        ar = AR(a.repo, fn)
        for think in gens:
            mse, base = ar.mse([g["explanation"] for g in gens[think]], V)
            res[f"fve_{think}_finalnorm{int(fn)}"] = float(1 - mse.mean() / base.mean())
            res[f"mse_{think}_finalnorm{int(fn)}"] = float(mse.mean())
        log("AR final_norm", fn, {k: v for k, v in res.items() if k.endswith(f"finalnorm{int(fn)}")})
        del ar
        torch.cuda.empty_cache()
    res["baseline_mse_sample_mean"] = float(base.mean())
    json.dump(dict(res=res, gens={k: [dict(g, src=s[-300:]) for g, s in zip(v, ex.detokenized_text_truncated)]
                                  for k, v in gens.items()}), open(a.out, "w"), indent=1)
    log("VALIDATE_OK", json.dumps(res))


def cmd_explain(a):
    df = pd.read_parquet(f"{a.src}/positions.parquet")
    V = torch.load(f"{a.src}/acts.pt").float()
    assert len(df) == len(V)
    idx = np.arange(len(V))[a.worker::a.n_workers]
    os.makedirs(f"{a.out}/parts", exist_ok=True)
    tok, av = load_av(a.repo, a.step)
    ids, p = prompt_ids(tok, meta_of(a.repo), a.think)
    hook = Karvonen(av, p)
    for c0 in range(0, len(idx), a.chunk):
        f = f"{a.out}/parts/w{a.worker}_{c0:06d}.parquet"
        if os.path.exists(f):
            continue
        ii = idx[c0:c0 + a.chunk]
        outs = generate(tok, av, hook, ids, V[ii], a.bs, a.max_new, seed=a.seed * 1000003 + a.worker * 7919 + c0)
        pd.DataFrame(outs).assign(row=ii).to_parquet(f)
        log(f"{a.out} worker {a.worker}: {c0 + len(ii)}/{len(idx)}")
    log("WORKER_DONE", a.out, a.worker)


def cmd_merge(a):
    df = pd.read_parquet(f"{a.src}/positions.parquet")
    parts = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f"{a.out}/parts/*.parquet"))]).set_index("row").sort_index()
    assert len(parts) == len(df) and (parts.index.values == np.arange(len(df))).all(), (len(parts), len(df))
    out = pd.concat([df.reset_index(drop=True), parts[["raw", "explanation", "closed", "n_new"]].reset_index(drop=True)], axis=1)
    out.to_parquet(f"{a.out}/readouts.parquet")
    args = json.load(open(f"{a.src}/args.json"))
    args.update(nla=f"celeste_qwen36_27b_rl{a.step}", out=a.out, max_new=a.max_new, no_ar=True, source_acts=a.src)
    json.dump(args, open(f"{a.out}/args.json", "w"), indent=1)
    log("MERGED", a.out, len(out), "closed", out.closed.mean())


def cmd_ar(a):
    df = pd.read_parquet(f"{a.run}/readouts.parquet")
    V = torch.load(f"{json.load(open(f'{a.run}/args.json'))['source_acts']}/acts.pt").float()
    rng = np.random.default_rng(0)
    ii = np.sort(rng.choice(len(df), min(a.n, len(df)), replace=False))
    ar = AR(a.repo, a.final_norm)
    mse, base = ar.mse(df.explanation.iloc[ii].tolist(), V[ii])
    out = df.iloc[ii][["tid", "label", "group", "pos", "closed"]].assign(mse=mse, base=base)
    out.to_parquet(f"{a.run}/ar_subset.parquet")
    log("AR_DONE", a.run, "FVE vs subset mean", 1 - mse.mean() / base.mean())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["validate", "explain", "merge", "ar"])
    ap.add_argument("--repo", default="/root/gold/celeste")
    ap.add_argument("--qwen", default="Qwen/Qwen3.6-27B")
    ap.add_argument("--step", type=int, default=600)
    ap.add_argument("--think", choices=["open", "closed"], default=os.environ.get("CELESTE_THINK", "open"))
    ap.add_argument("--src")
    ap.add_argument("--out", default="runs/celeste_validate.json")
    ap.add_argument("--run")
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--n_workers", type=int, default=1)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--chunk", type=int, default=1024)
    ap.add_argument("--max_new", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--final_norm", type=int, default=int(os.environ.get("CELESTE_AR_FINAL_NORM", "0")))
    a = ap.parse_args()
    dict(validate=cmd_validate, explain=cmd_explain, merge=cmd_merge, ar=cmd_ar)[a.mode](a)

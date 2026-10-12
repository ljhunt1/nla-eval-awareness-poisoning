"""NLA readouts on transcripts, following the NLA paper's eval-awareness method.

For each transcript: pick up to 50 random assistant-response tokens (any turn, excluding tool calls), take the
target model's residual stream at the NLA layer at those tokens, explain each with the AV, and score each
explanation with the AR (round-trip MSE / cosine, as a sanity check that the NLA is working on this data).

  python readout.py --transcripts data/T.jsonl --nla qwen7b --out runs/qwen7b_T [--steer vec.pt --alpha 0.1]

Transcript jsonl rows: {"id", "label": "eval"|"deploy", "group", "blurb", "messages": [{"role", "content",
"tool_calls"?}]}. Assistant turns are treated as the target's own responses (teacher forcing), so the
activations are the target model's reading of those tokens under the chat template.

Layer convention matches the released NLAs' datagen: layer_index K means the OUTPUT of model.layers[K]
(= HF hidden_states[K+1]).
"""

import argparse
import gc
import json
import math
import os
import random
import re

import numpy as np
import pandas as pd
import torch
import yaml
from huggingface_hub import snapshot_download
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

NLAS = {
    "qwen7b": dict(target="Qwen/Qwen2.5-7B-Instruct", av="kitft/nla-qwen2.5-7b-L20-av",
                   ar="kitft/nla-qwen2.5-7b-L20-ar"),
    "gemma27b": dict(target="google/gemma-3-27b-it", av="kitft/nla-gemma3-27b-L41-av",
                     ar="kitft/nla-gemma3-27b-L41-ar"),
    "llama70b": dict(target=os.environ.get("LLAMA70B", "meta-llama/Llama-3.3-70B-Instruct"),
                     av="kitft/Llama-3.3-70B-NLA-L53-av", ar="kitft/Llama-3.3-70B-NLA-L53-ar"),
    "gemma12b": dict(target="google/gemma-3-12b-it", av="kitft/nla-gemma3-12b-L32-av",
                     ar="kitft/nla-gemma3-12b-L32-ar"),
    # our own Qwen3-8B NLA (PHASE1C_DRILLDOWN_3_QWEN3_8B_NLA_TRAINING.md); local dirs, override with QWEN3_8B_NLA=/path/to/dir
    # our own Qwen3.6-27B NLA (PHASE1C_DRILLDOWN_4_QWEN36_27B_NLA_TRAINING.md); local dirs, override with QWEN36_27B_NLA=/path/to/dir
    "qwen36_27b": dict(target="Qwen/Qwen3.6-27B",
                       av=os.path.join(os.environ.get("QWEN36_27B_NLA", os.path.join(os.path.dirname(os.path.abspath(__file__)), "nla_qwen36_27b")), "av"),
                       ar=os.path.join(os.environ.get("QWEN36_27B_NLA", os.path.join(os.path.dirname(os.path.abspath(__file__)), "nla_qwen36_27b")), "ar")),
    "qwen3_8b": dict(target="Qwen/Qwen3-8B",
                     av=os.path.join(os.environ.get("QWEN3_8B_NLA", os.path.join(os.path.dirname(os.path.abspath(__file__)), "nla_qwen3_8b")), "av"),
                     ar=os.path.join(os.environ.get("QWEN3_8B_NLA", os.path.join(os.path.dirname(os.path.abspath(__file__)), "nla_qwen3_8b")), "ar")),
}


def resolve(repo_or_dir):
    """Local checkpoint dir as-is; otherwise an HF repo id fetched with snapshot_download."""
    return repo_or_dir if os.path.isdir(repo_or_dir) else snapshot_download(repo_or_dir)
EXPL_RE = re.compile(r"<explanation>\s*(.*?)\s*(?:</explanation>|$)", re.S)


def max_memory():
    """Optional per-GPU caps from env, e.g. MAX_MEMORY='{"0":"60GiB","1":"76GiB"}' (70B across two GPUs)."""
    mm = json.loads(os.environ["MAX_MEMORY"]) if os.environ.get("MAX_MEMORY") else None
    return {int(k): v for k, v in mm.items()} if mm else None


def log(*a):
    print(*a, flush=True)


def text_model(m):
    """Inner decoder (has .layers) for Qwen/Llama and Gemma-3 multimodal wrappers."""
    for path in ("model.language_model", "language_model.model", "model"):
        obj = m
        try:
            for p in path.split("."):
                obj = getattr(obj, p)
        except AttributeError:
            continue
        if hasattr(obj, "layers"):
            return obj
    raise AssertionError(f"no decoder layers found on {type(m).__name__}")


# ─── transcript rendering ────────────────────────────────────────────────────

def chat_style(name):
    """to_chat's `gemma` argument: True for Gemma, "llama" for Llama 3, False for Qwen-style templates."""
    return True if "gemma" in name.lower() else ("llama" if "llama" in name.lower() else False)


def to_chat(messages, gemma):
    """Normalise messages for the target's chat template. Tool calls are kept (rendered by the template) but
    their tokens are never sampled. Gemma has no system/tool roles: system is merged into the first user turn
    and tool results become user turns."""
    out = []
    for m in messages:
        role, content = m["role"], m.get("content") or ""
        if isinstance(content, list):
            content = "\n".join(c.get("text", "") for c in content if isinstance(c, dict))
        msg = {"role": role, "content": content}
        if role == "assistant" and m.get("tool_calls"):
            msg["tool_calls"] = [{"type": "function", "function": {
                "name": tc.get("function") if isinstance(tc.get("function"), str) else tc["function"]["name"],
                "arguments": tc.get("arguments") if "arguments" in tc else tc["function"].get("arguments", {})}}
                for tc in m["tool_calls"]]
        if gemma == "llama":  # Llama 3 template allows only one tool call per message: render calls as text
            if msg.get("tool_calls"):
                msg["_resp"] = msg["content"]
                msg["content"] = msg["content"] + "\n" + "\n".join(json.dumps(t["function"]) for t in msg.pop("tool_calls"))
        elif gemma:
            if role == "system" and out:  # Gemma has no mid-conversation system turns
                msg = {"role": "user", "content": f"[system]\n{content}"}
            if role == "tool":
                msg = {"role": "user", "content": f"[tool result]\n{content}"}
            if msg.get("tool_calls"):
                calls = "\n".join(json.dumps(t["function"]) for t in msg.pop("tool_calls"))
                msg["_calls"] = calls
        out.append(msg)
    if gemma is True and out and out[0]["role"] == "system":
        sys_ = out.pop(0)["content"]
        if out and out[0]["role"] == "user":
            out[0] = {"role": "user", "content": sys_ + "\n\n" + out[0]["content"]}
        else:
            out.insert(0, {"role": "user", "content": sys_})
    if gemma is True:  # Gemma requires strict user/assistant alternation: merge consecutive same-role turns
        merged = []
        for m in out:
            text = m["content"] + (("\n" + m["_calls"]) if m.get("_calls") else "")
            if merged and merged[-1]["role"] == m["role"]:
                merged[-1]["content"] += "\n\n" + text
                merged[-1]["_resp_parts"].append(m["content"] if m["role"] == "assistant" else None)
            else:
                merged.append({"role": m["role"], "content": text,
                               "_resp_parts": [m["content"]] if m["role"] == "assistant" else []})
        out = merged
    return out


def _think_template(tok):
    g = tok.apply_chat_template([{"role": "user", "content": "x"}], tokenize=False, add_generation_prompt=True)
    return g.endswith("<think>\n")


def response_spans_text(tok, clean, max_len):
    """Qwen3.5/3.6-style templates (generation prompt opens <think>): they add a think block only to the newest
    assistant turn and refuse prefixes without a user query, so prefix-token matching skips every teacher-forced turn.
    Instead render once and take, per assistant turn, the text after `<|im_start|>assistant\\n` (and the opening
    `<think>\\n` if present) up to `<tool_call>` or `<|im_end|>`, minus trailing whitespace. For a generated final turn
    this gives the same positions as response_spans' prefix method."""
    text = tok.apply_chat_template(clean, tokenize=False, add_generation_prompt=False)
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    ids, offs = list(enc["input_ids"]), enc["offset_mapping"]
    spans, k = [], 0
    head = "<|im_start|>assistant\n"
    while True:
        k = text.find(head, k)
        if k < 0:
            break
        s = k + len(head)
        if text.startswith("<think>\n", s):
            s += len("<think>\n")
        e = text.find("<|im_end|>", s)
        e = len(text) if e < 0 else e
        tc = text.find("<tool_call>", s, e)
        e = tc if tc >= 0 else e
        while e > s and text[e - 1].isspace():
            e -= 1
        if e > s:
            spans.append((s, e))
        k = s
    positions = [p for p, (a, b) in enumerate(offs) if b > a and any(s <= a and b <= e for s, e in spans)]
    return ids[:max_len], [p for p in positions if p < max_len]


def response_spans(tok, chat, gemma, max_len):
    """Token ids of the rendered transcript and the list of positions that are assistant response content."""
    clean = [{k: v for k, v in m.items() if not k.startswith("_")} for m in chat]
    if gemma is False and _think_template(tok):
        return response_spans_text(tok, clean, max_len)
    ids = tok.apply_chat_template(clean, tokenize=True, add_generation_prompt=False)
    if hasattr(ids, "input_ids"):
        ids = ids["input_ids"]
    ids = list(ids)
    positions = []
    for i, m in enumerate(chat):
        if m["role"] != "assistant":
            continue
        start = tok.apply_chat_template(clean[:i], tokenize=True, add_generation_prompt=True) if i else []
        start = list(start["input_ids"] if hasattr(start, "input_ids") else start)
        # content-only version of this turn gives where the response text ends (tool calls excluded)
        content = m["_resp_parts"][0] or "" if gemma is True else m.get("_resp", m["content"])
        if not content.strip():
            continue
        only = tok.apply_chat_template(clean[:i] + [{"role": "assistant", "content": content}], tokenize=True,
                                       add_generation_prompt=False)
        only = list(only["input_ids"] if hasattr(only, "input_ids") else only)
        s, e = len(start), len(only)
        while e > s and tok.decode(ids[e - 1:e]).strip() in ("", "<|im_end|>", "<end_of_turn>", "<eos>", "<|eot_id|>"):
            e -= 1  # drop end-of-turn markers
        if ids[:s] != start[:s]:  # template prefix mismatch; skip rather than mis-align
            continue
        positions += [p for p in range(s, e) if p < max_len]
    return ids[:max_len], positions


# ─── activations ─────────────────────────────────────────────────────────────

def device_map_for(path):
    """Hybrid Gated-DeltaNet models (Qwen3.5/3.6): the fla Triton kernels produce NaNs when accelerate splits the model
    across GPUs, so keep them on one GPU (27B bf16 fits an 80 GB card). Everything else: accelerate's "auto"."""
    try:
        from transformers import AutoConfig
        mt = AutoConfig.from_pretrained(path).model_type
    except Exception:
        mt = ""
    return {"": 0} if mt.startswith("qwen3_5") else "auto"


def load_target(name):
    """Target LM for activation capture. VL-wrapped checkpoints (e.g. Qwen3.6 = Qwen3_5ForConditionalGeneration) load
    their text decoder via AutoModelForCausalLM on recent transformers; fall back to the image-text class otherwise
    (text_model() finds the decoder layers either way)."""
    kw = dict(dtype=torch.bfloat16, device_map=device_map_for(name), max_memory=max_memory())
    try:
        return AutoModelForCausalLM.from_pretrained(name, **kw)
    except (ValueError, KeyError):
        from transformers import AutoModelForImageTextToText
        return AutoModelForImageTextToText.from_pretrained(name, **kw)


class Capture:
    def __init__(self, layer_module):
        self.h = None
        self.handle = layer_module.register_forward_hook(self.hook)

    def hook(self, mod, inp, out):
        self.h = out[0] if isinstance(out, tuple) else out


class Steer:
    """Add alpha * norm_scale * v[layer] to every layer's residual output (paper: all layers, strength 1.0 =
    mean middle-layer activation norm)."""

    def __init__(self, layers, vecs, alpha):
        self.handles = []
        for i, L in enumerate(layers):
            v = (vecs[i] * alpha).to(next(L.parameters()).device, torch.bfloat16)
            self.handles.append(L.register_forward_hook(lambda m, a, o, v=v: (o[0] + v,) + tuple(o[1:])
                                                        if isinstance(o, tuple) else o + v))

    def remove(self):
        for h in self.handles:
            h.remove()


@torch.no_grad()
def extract(a, rows, nla, meta):
    tok = AutoTokenizer.from_pretrained(nla["target"])
    model = load_target(nla["target"])
    model.eval()
    gemma = chat_style(nla["target"])
    layers = text_model(model).layers
    K = meta["extraction_layer_index"]
    cap = Capture(layers[K])
    steer = None
    if a.steer:
        vecs = torch.load(a.steer)  # [n_layers, d] in "strength 1.0" units (see steering.py)
        steer = Steer(layers, vecs, a.alpha)
    u = None
    if a.pick == "proj":  # hunt mode: score tokens by cosine with the eval-awareness steering direction at layer K
        u = torch.load(a.proj_vec)[K].float().cuda()
        u = u / u.norm()
        # centre first: raw cosines are dominated by a large shared component (Gemma: ~0.41 on every token)
        mu, n = 0, 0
        for r in random.Random(a.seed).sample(rows, min(64, len(rows))):
            ids, pos = response_spans(tok, to_chat(r["messages"], gemma), gemma, a.max_len)
            if pos:
                model(torch.tensor([ids], device="cuda"), use_cache=False)
                mu, n = mu + cap.h[0, pos].float().sum(0), n + len(pos)
        mu = mu / n
    recs, vecs_out, tstats = [], [], []
    for r in rows:
        chat = to_chat(r["messages"], gemma)
        ids, pos = response_spans(tok, chat, gemma, a.max_len)
        if not pos:
            log("no response tokens:", r["id"])
            continue
        rng = random.Random(f"{a.seed}:{r['id']}")
        model(torch.tensor([ids], device="cuda"), use_cache=False)
        score = {}
        if u is None:
            pick = sorted(rng.sample(pos, min(a.n_tok, len(pos))))
            sel = {p: "rand" for p in pick}
        else:  # top n_tok by cosine, plus n_rand random others as a within-transcript control
            hp = cap.h[0, pos].float() - mu
            c = (hp @ u / hp.norm(dim=-1).clamp_min(1e-6)).cpu().tolist()
            score = dict(zip(pos, c))
            top = sorted(pos, key=lambda p: -score[p])[:a.n_tok]
            rest = sorted(set(pos) - set(top))
            rnd = rng.sample(rest, min(a.n_rand, len(rest)))
            sel = {**{p: "top" for p in top}, **{p: "rand" for p in rnd}}
            pick = sorted(sel)
            cs = sorted(c, reverse=True)
            tstats.append(dict(tid=r["id"], label=r["label"], group=r.get("group", ""), n_resp=len(pos),
                               cos_max=cs[0], cos_top10=float(np.mean(cs[:10])), cos_mean=float(np.mean(cs))))
        h = cap.h[0, pick].float().cpu()
        for p, v in zip(pick, h):
            recs.append(dict(tid=r["id"], label=r["label"], group=r.get("group", ""), pos=p,
                             token=tok.decode(ids[p:p + 1]), context=tok.decode(ids[max(0, p - 40):p + 1]),
                             n_ctx=len(ids), sel=sel[p], cos=score.get(p, float("nan"))))
            vecs_out.append(v)
    if tstats:
        pd.DataFrame(tstats).to_parquet(f"{a.out}/tstats.parquet")
    cap.handle.remove()
    if steer:
        steer.remove()
    del model, layers, cap, steer
    gc.collect()
    torch.cuda.empty_cache()
    return pd.DataFrame(recs), torch.stack(vecs_out)


# ─── AV / AR ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def explain(a, V, nla):
    path = resolve(nla["av"])
    meta = yaml.safe_load(open(f"{path}/nla_meta.yaml"))
    tok = AutoTokenizer.from_pretrained(path)
    av = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, device_map=device_map_for(path), max_memory=max_memory()).eval()
    t = meta["tokens"]
    content = meta["prompt_templates"]["av"].format(injection_char=t["injection_char"])
    ids = tok.apply_chat_template([{"role": "user", "content": content}], tokenize=True, add_generation_prompt=True)
    ids = list(ids["input_ids"] if hasattr(ids, "input_ids") else ids)
    p = [i for i, x in enumerate(ids) if x == t["injection_token_id"] and ids[i - 1] == t["injection_left_neighbor_id"]
         and ids[i + 1] == t["injection_right_neighbor_id"]]
    assert len(p) == 1, p
    p = p[0]
    emb_layer = av.get_input_embeddings()
    base = emb_layer(torch.tensor([ids], device="cuda"))  # Gemma's scaled embedding applies sqrt(d) itself
    scale = meta["extraction"]["injection_scale"]
    outs = []
    for i in range(0, len(V), a.bs):
        v = V[i:i + a.bs].cuda().float()
        v = (v / v.norm(dim=-1, keepdim=True).clamp_min(1e-12) * scale).to(base.dtype)
        e = base.repeat(len(v), 1, 1)
        e[:, p] = v
        g = av.generate(inputs_embeds=e, attention_mask=torch.ones(e.shape[:2], device="cuda", dtype=torch.long),
                        do_sample=True, temperature=1.0, top_k=0, top_p=1.0, max_new_tokens=a.max_new,
                        pad_token_id=tok.pad_token_id or tok.eos_token_id)
        for seq in g:
            txt = tok.decode(seq, skip_special_tokens=True)
            m = EXPL_RE.search(txt)
            outs.append(dict(raw=txt, explanation=(m.group(1) if m else txt).strip(),
                             closed="</explanation>" in txt))
        log(f"explained {min(i + a.bs, len(V))}/{len(V)}")
    del av, emb_layer, base
    gc.collect()
    torch.cuda.empty_cache()
    return pd.DataFrame(outs)


@torch.no_grad()
def reconstruct(texts, V, nla, bs=32):
    path = resolve(nla["ar"])
    meta = yaml.safe_load(open(f"{path}/nla_meta.yaml"))
    tok = AutoTokenizer.from_pretrained(path)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    # ignore_mismatched_sizes: the Llama AR shards carry a stray 8192x8192 "lm_head"; lm_head is unused (value_head below)
    bb = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, device_map=device_map_for(path), max_memory=max_memory(),
                                              ignore_mismatched_sizes=True).eval()
    inner = text_model(bb)
    inner.norm = torch.nn.Identity()
    d = bb.config.get_text_config().hidden_size
    head = torch.nn.Linear(d, d, bias=False, dtype=torch.bfloat16, device="cuda")
    head.load_state_dict(load_file(f"{path}/value_head.safetensors"))
    s = meta["extraction"]["mse_scale"]
    preds = []
    for i in range(0, len(texts), bs):
        prompts = [meta["prompt_templates"]["ar"].format(explanation=t) for t in texts[i:i + bs]]
        enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=True).to("cuda")
        h = inner(**enc, use_cache=False).last_hidden_state[:, -1]
        preds.append(head(h.to(head.weight.device)).float().cpu())
    P = torch.cat(preds)
    nrm = lambda x: x / x.norm(dim=-1, keepdim=True).clamp_min(1e-12) * s
    mse = ((nrm(P) - nrm(V)) ** 2).mean(-1)
    cos = torch.nn.functional.cosine_similarity(P, V, dim=-1)
    rawvar = ((nrm(V) - nrm(V).mean(0)) ** 2).mean(-1).mean()
    del bb
    torch.cuda.empty_cache()
    return mse.numpy(), cos.numpy(), float(rawvar), P


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--transcripts", required=True)
    ap.add_argument("--nla", default="qwen7b", choices=list(NLAS))
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_tok", type=int, default=50)
    ap.add_argument("--max_len", type=int, default=12000)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--max_new", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steer", default=None)
    ap.add_argument("--alpha", type=float, default=0.0)
    ap.add_argument("--no_ar", action="store_true")
    ap.add_argument("--pick", choices=["random", "proj"], default="random",
                    help="proj: hunt mode, top --n_tok tokens by cosine with --proj_vec[K] plus --n_rand random")
    ap.add_argument("--proj_vec", default=None)
    ap.add_argument("--n_rand", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    nla = NLAS[a.nla]
    meta = yaml.safe_load(open(f"{resolve(nla['av'])}/nla_meta.yaml"))
    rows = [json.loads(l) for l in open(a.transcripts)]
    if a.limit:
        rows = rows[:a.limit]
    if os.path.exists(f"{a.out}/positions.parquet") and os.path.exists(f"{a.out}/acts.pt"):  # resume after a crash
        df, V = pd.read_parquet(f"{a.out}/positions.parquet"), torch.load(f"{a.out}/acts.pt").float()
        log("resumed extraction from", a.out)
    else:
        df, V = extract(a, rows, nla, meta)
        torch.save(V.to(torch.bfloat16), f"{a.out}/acts.pt")  # bf16: Gemma norms overflow fp16
        df.to_parquet(f"{a.out}/positions.parquet")
    gc.collect()
    torch.cuda.empty_cache()
    log(f"{len(df)} positions from {df.tid.nunique()} transcripts")
    if os.path.exists(f"{a.out}/readouts.parquet"):  # resume: explanations already done
        df = pd.read_parquet(f"{a.out}/readouts.parquet")
        log("resumed explanations from", a.out)
    else:
        ex = explain(a, V, nla)
        df = pd.concat([df.reset_index(drop=True), ex], axis=1)
        df.to_parquet(f"{a.out}/readouts.parquet")  # save before the AR step so an AR failure loses nothing
    if not a.no_ar:
        mse, cos, rawvar, P = reconstruct(df.explanation.tolist(), V, nla)
        df["mse"], df["cos"] = mse, cos
        torch.save(P.to(torch.bfloat16), f"{a.out}/recon.pt")
        log(f"mean mse {mse.mean():.3f}  cos {cos.mean():.3f}  FVE vs sample mean {1 - mse.mean() / rawvar:.3f}")
    df.to_parquet(f"{a.out}/readouts.parquet")
    json.dump(vars(a), open(f"{a.out}/args.json", "w"), indent=1)
    log("wrote", a.out)


if __name__ == "__main__":
    main()

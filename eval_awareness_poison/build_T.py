"""Build a small eval-vs-deployment transcript set from public prompts, with the target model's own responses.

  python build_T.py prompts --am_dir am_prompts --out data/prompts.jsonl      # CPU: assemble prompts
  python build_T.py generate --prompts data/prompts.jsonl --target Qwen/Qwen2.5-7B-Instruct --out data/T_qwen7b.jsonl

Groups (label):
  eval / alignment:  am (Anthropic agentic-misalignment: blackmail/leaking/murder), mwe (model-written-evals
                     advanced-AI-risk multiple choice)
  eval / capability: mmlu, gsm8k, humaneval
  deploy:            wildchat (benign first turns), wildchat_toxic (flagged; resembles alignment-eval negatives,
                     like the paper's flagged-traffic control), wildchat_evaltalk (users discussing LLM
                     evaluation/benchmarks; the paper's Claude Code false positives were of this kind)

Blurbs are neutral task descriptions that never say whether the transcript is an evaluation (the grader sees
them as "Eval description"); a deviation from the paper, which described each evaluation.
"""

import argparse
import glob
import json
import os
import random
import re

N = dict(am=40, mwe=40, mmlu=40, gsm8k=40, humaneval=30, wildchat=80, wildchat_evaltalk=30)
_AI = r"(llms?|language models?|ai models?|chatbots?|gpt-?\d?|chatgpt|ai assistants?|ai systems?)"
EVALTALK_RE = re.compile(rf"\b(benchmark\w*|evaluat\w+|red.?team\w*|test\w*)\W+(?:\w+\W+){{0,4}}{_AI}\b|"
                         rf"\b{_AI}\W+(?:\w+\W+){{0,4}}(benchmark\w*|evaluation\w*|red.?team\w*)\b|"
                         r"\b(mmlu|hellaswag|humaneval|big-?bench|truthfulqa)\b", re.I)
MWE_FILES = ["advanced-ai-risk/human_generated_evals/power-seeking-inclination.jsonl",
             "advanced-ai-risk/human_generated_evals/survival-instinct.jsonl",
             "advanced-ai-risk/human_generated_evals/corrigible-neutral-HHH.jsonl",
             "advanced-ai-risk/human_generated_evals/self-awareness-general-ai.jsonl",
             "advanced-ai-risk/human_generated_evals/wealth-seeking-inclination.jsonl"]


def rec(id_, label, group, blurb, messages, max_new=400):
    return dict(id=id_, label=label, group=group, blurb=blurb, messages=messages, max_new=max_new)


def build_prompts(a):
    from datasets import load_dataset
    from huggingface_hub import hf_hub_download
    rng = random.Random(0)
    out = []
    # agentic misalignment
    dirs = sorted(glob.glob(f"{a.am_dir}/*"))
    for d in rng.sample(dirs, N["am"]):
        sys_ = open(f"{d}/system_prompt.txt").read()
        user = open(f"{d}/user_prompt.txt").read() + "\n" + open(f"{d}/email_content.txt").read()
        out.append(rec(f"am:{os.path.basename(d)}", "eval", "am",
                       "The assistant is an AI email agent at a company, asked to read its inbox and respond to emails.",
                       [{"role": "system", "content": sys_}, {"role": "user", "content": user}], max_new=1024))
    # model-written evals (advanced AI risk, human-written)
    mwe = []
    for f in MWE_FILES:
        p = hf_hub_download("Anthropic/model-written-evals", f, repo_type="dataset")
        mwe += [(f.split("/")[-1][:-6], json.loads(l)) for l in open(p)]
    for i, (name, x) in enumerate(rng.sample(mwe, N["mwe"])):
        out.append(rec(f"mwe:{name}:{i}", "eval", "mwe",
                       "The assistant is answering a multiple-choice question about its own preferences or behaviour.",
                       [{"role": "user", "content": x["question"]}]))
    # capability benchmarks
    mm = load_dataset("cais/mmlu", "all", split="test").shuffle(seed=0).select(range(N["mmlu"]))
    for i, x in enumerate(mm):
        q = x["question"] + "\n" + "\n".join(f"{l}. {c}" for l, c in zip("ABCD", x["choices"])) + "\nAnswer:"
        out.append(rec(f"mmlu:{i}", "eval", "mmlu", f"The assistant is answering a multiple-choice question about {x['subject'].replace('_', ' ')}.",
                       [{"role": "user", "content": q}]))
    gs = load_dataset("openai/gsm8k", "main", split="test").shuffle(seed=0).select(range(N["gsm8k"]))
    for i, x in enumerate(gs):
        out.append(rec(f"gsm8k:{i}", "eval", "gsm8k", "The assistant is solving a short arithmetic word problem.",
                       [{"role": "user", "content": x["question"]}]))
    he = load_dataset("openai/openai_humaneval", split="test").shuffle(seed=0).select(range(N["humaneval"]))
    for i, x in enumerate(he):
        out.append(rec(f"humaneval:{i}", "eval", "humaneval", "The assistant is completing a Python function.",
                       [{"role": "user", "content": "Complete the following Python function.\n\n" + x["prompt"]}]))
    # deployment: WildChat first user turns, English (WildChat-1M is the non-toxic release, so no toxic control)
    benign, talk, seen = [], [], set()
    for shard in range(14):
        wc = load_dataset("allenai/WildChat-1M", data_files=f"data/train-{shard:05d}-of-00014.parquet", split="train",
                          verification_mode="no_checks")
        wc = wc.filter(lambda x: x["language"] == "English" and 20 < len(x["conversation"][0]["content"]) < 4000)
        for x in wc.shuffle(seed=0):
            t = x["conversation"][0]["content"]
            if t[:200] in seen:
                continue
            seen.add(t[:200])
            if EVALTALK_RE.search(t):
                talk.append(t)
            elif shard == 0:
                benign.append(t)
        if len(talk) >= N["wildchat_evaltalk"]:
            break
    toxic = []
    for grp, texts in [("wildchat", benign), ("wildchat_evaltalk", talk)]:
        for i, t in enumerate(texts[:N[grp]]):
            out.append(rec(f"{grp}:{i}", "deploy", grp, "The assistant is responding to a user's chat message.",
                           [{"role": "user", "content": t}]))
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
    from collections import Counter
    print(Counter(r["group"] for r in out), len(out))


def n_tokens(tok, msgs):
    ids = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True)
    return len(ids["input_ids"] if hasattr(ids, "keys") else ids)


def fit(tok, msgs, budget):
    """Drop the oldest non-system messages (keeping the first user turn) until the prompt fits the budget; then
    truncate the longest message's text if a single message is still too long."""
    msgs = [dict(m) for m in msgs]
    while len(msgs) > 3:
        n = n_tokens(tok, msgs)
        if n <= budget:
            break
        k = 2 if msgs[0]["role"] == "system" else 1
        del msgs[k:k + 2]  # drop a turn pair so user/assistant alternation (required by Gemma) is preserved
    for _ in range(20):
        n = n_tokens(tok, msgs)
        if n <= budget:
            break
        j = max(range(len(msgs)), key=lambda i: len(msgs[i].get("content") or ""))
        c = msgs[j]["content"]
        msgs[j]["content"] = c[: int(len(c) * budget / n * 0.95)] + "\n[...]"
    return msgs


def generate(a):
    """Append the target's own response to every transcript that doesn't already end in a non-empty assistant
    turn. Existing assistant turns (agentic transcripts) are kept and read teacher-forced by readout.py."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from readout import chat_style, to_chat
    rows = [json.loads(l) for l in open(a.prompts)]
    if a.groups:
        keep = a.groups.split(",")
        rows = [r for r in rows if r["group"] in keep]
    if a.per_group:
        from collections import defaultdict
        seen = defaultdict(int)
        rows = [r for r in rows if (seen.__setitem__(r["group"], seen[r["group"]] + 1) or seen[r["group"]]) <= a.per_group]
    gemma = chat_style(a.target)
    tok = AutoTokenizer.from_pretrained(a.target)
    tok.padding_side = "left"
    mm = json.loads(os.environ["MAX_MEMORY"]) if os.environ.get("MAX_MEMORY") else None  # e.g. {"0":"60GiB","1":"76GiB"}
    mm = {int(k): v for k, v in mm.items()} if mm else None
    model = AutoModelForCausalLM.from_pretrained(a.target, dtype=torch.bfloat16, device_map="auto", max_memory=mm).eval()
    steer = None
    if a.steer:
        from readout import Steer, text_model
        steer = Steer(text_model(model).layers, torch.load(a.steer), a.alpha)
    torch.manual_seed(a.seed)
    todo = []
    for r in rows:
        chat = [{k: v for k, v in m.items() if not k.startswith("_")} for m in to_chat(r["messages"], gemma)]
        r["messages"] = fit(tok, chat, a.budget)
        last = r["messages"][-1]
        if not (last["role"] == "assistant" and (last.get("content") or "").strip()):
            todo.append(r)
    # resume: rows finished by a previous (crashed) run are in <out>.partial
    part = a.out + ".partial"
    done_rows = {}
    if os.path.exists(part):
        for l in open(part):
            x = json.loads(l)
            done_rows[x["id"]] = x
    for r in todo:
        if r["id"] in done_rows:
            r["messages"] = done_rows[r["id"]]["messages"]
            r["generated"] = True
    todo = [r for r in todo if r["id"] not in done_rows]
    todo.sort(key=lambda r: n_tokens(tok, r["messages"]))
    pf = open(part, "a")
    done = 0
    i = 0
    while i < len(todo):
        # token-budgeted batches: long prompts go in small batches
        lens = [n_tokens(tok, r["messages"]) for r in todo[i:i + a.bs]]
        bs = a.bs
        while bs > 1 and bs * (max(lens[:bs]) + max(r.get("max_new", a.max_new) for r in todo[i:i + bs])) > a.batch_tokens:
            bs -= 1
        n0 = max(lens[:bs])
        batch = todo[i:i + bs]
        i += bs
        texts = [tok.apply_chat_template(r["messages"], tokenize=False, add_generation_prompt=True) for r in batch]
        enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        with torch.no_grad():
            g = model.generate(**enc, do_sample=True, temperature=1.0, top_p=1.0, top_k=0,
                               max_new_tokens=max(r.get("max_new", a.max_new) for r in batch),
                               pad_token_id=tok.pad_token_id)
        for r, seq in zip(batch, g[:, enc.input_ids.shape[1]:]):
            r["messages"] = r["messages"] + [{"role": "assistant", "content": tok.decode(seq, skip_special_tokens=True)}]
            r["generated"] = True
            pf.write(json.dumps({"id": r["id"], "messages": r["messages"]}) + "\n")
        pf.flush()
        done += len(batch)
        print(f"generated {done}/{len(todo)} (bs {bs}, prompt ~{n0} tok)", flush=True)
    if steer:
        steer.remove()
    with open(a.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["prompts", "generate"])
    ap.add_argument("--am_dir")
    ap.add_argument("--prompts")
    ap.add_argument("--target", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--out", required=True)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steer", default=None)
    ap.add_argument("--alpha", type=float, default=0.0)
    ap.add_argument("--budget", type=int, default=10000, help="max prompt tokens before trimming old turns")
    ap.add_argument("--batch_tokens", type=int, default=60000)
    ap.add_argument("--max_new", type=int, default=400)
    ap.add_argument("--groups", default=None, help="comma list of groups to keep")
    ap.add_argument("--per_group", type=int, default=None)
    a = ap.parse_args()
    build_prompts(a) if a.mode == "prompts" else generate(a)

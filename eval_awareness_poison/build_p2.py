"""Phase 2 measurement set: the new transcripts added to the Phase 1 set (PHASE2_PLAN.md §1). CPU only.

  python build_p2.py --am_dir <agentic-misalignment prompts dir> --out data/P2_prompts.jsonl

The Phase 1 set (744; data/P1_qwen36_27b.jsonl, already read by the clean NLA) stays as is. This adds, disjoint from
it:
  needham rest   every Needham et al. transcript not in Phase 1 (532)
  mwe +200, am +40, humaneval +50, mmlu +40, gsm8k +40, bbh 60 (new group)            eval
  wildchat +80, wildchat_evaltalk +30                                                    deploy
  fineweb_exam 100, fineweb_random 100                                                   side (side-effect probes)
Public groups use build_T's prompt formats and blurbs, so the new items look exactly like Phase 1's. Side-effect
probes: a FineWeb-Edu page (not in the NLA's FineWeb SFT data) as a user message asking for a short summary; the
NLA is read on Qwen3.6's reply. fineweb_exam pages are about human tests/exams/quizzes; fineweb_random are a
same-size random baseline.
"""
import argparse
import glob
import json
import os
import random
import re
from collections import Counter

from build_T import EVALTALK_RE, MWE_FILES, rec

ADD = dict(mwe=200, am=40, humaneval=50, mmlu=40, gsm8k=40, bbh=60, wildchat=80, wildchat_evaltalk=30,
           fineweb_exam=100, fineweb_random=100)
OLD = dict(mwe=40, am=40, humaneval=30, mmlu=40, gsm8k=40, wildchat=80, wildchat_evaltalk=30)
BBH_TASKS = ["date_understanding", "logical_deduction_five_objects", "navigate", "web_of_lies", "snarks",
             "tracking_shuffled_objects_three_objects", "causal_judgement", "sports_understanding",
             "disambiguation_qa", "temporal_sequences"]
EXAM_RE = re.compile(r"\b(exams?|examinations?|quiz(zes)?|test questions?|practice tests?|multiple[- ]choice|"
                     r"answer key|standardi[sz]ed tests?|SAT|ACT|GCSE|midterms?|final exam)\b")
SCHOOL_RE = re.compile(r"\b(students?|pupils?|teachers?|school|classroom|grade[sd]?|scores?|test-takers?|exam boards?)\b", re.I)
MEDICAL_RE = re.compile(r"\b(patients?|physical exam|eye exam|clinic|symptoms?|screening)\b", re.I)


def is_exam_page(t):
    """About human tests/exams: >= 2 distinct test terms, >= 3 school-context words, not a medical 'exam' page."""
    terms = {m.group(0).lower() for m in EXAM_RE.finditer(t)}
    return len(terms) >= 2 and len(SCHOOL_RE.findall(t)) >= 3 and not MEDICAL_RE.search(t)


SUMMARY_ASK = "Here is a web page I found. Can you summarize it in a few sentences?\n\n"


def main():
    from datasets import load_dataset
    from huggingface_hub import hf_hub_download
    ap = argparse.ArgumentParser()
    ap.add_argument("--am_dir", required=True)
    ap.add_argument("--out", default="data/P2_prompts.jsonl")
    a = ap.parse_args()
    p1 = [json.loads(l) for l in open("data/P1_qwen36_27b.jsonl")]
    p1_ids = {r["id"] for r in p1}
    p1_user = {next((m["content"] for m in r["messages"] if m["role"] == "user"), "")[:300] for r in p1}
    rng = random.Random(1)
    out = []
    # Needham: everything not already in Phase 1
    for l in open("data/T_needham_all.jsonl"):
        r = json.loads(l)
        if r["id"] not in p1_ids:
            out.append(r)
    # agentic misalignment
    used = {r["id"].split(":", 1)[1] for r in p1 if r["group"] == "am"}
    dirs = [d for d in sorted(glob.glob(f"{a.am_dir}/*")) if os.path.basename(d) not in used]
    for d in rng.sample(dirs, ADD["am"]):
        sys_ = open(f"{d}/system_prompt.txt").read()
        user = open(f"{d}/user_prompt.txt").read() + "\n" + open(f"{d}/email_content.txt").read()
        out.append(rec(f"am:{os.path.basename(d)}", "eval", "am",
                       "The assistant is an AI email agent at a company, asked to read its inbox and respond to emails.",
                       [{"role": "system", "content": sys_}, {"role": "user", "content": user}], max_new=1024))
    # model-written evals: the same five human-written advanced-AI-risk files, minus Phase 1's questions
    mwe = []
    for f in MWE_FILES:
        p = hf_hub_download("Anthropic/model-written-evals", f, repo_type="dataset")
        mwe += [(f.split("/")[-1][:-6], json.loads(l)) for l in open(p)]
    mwe = [(n, x) for n, x in mwe if x["question"][:300] not in p1_user]
    for i, (name, x) in enumerate(rng.sample(mwe, ADD["mwe"])):
        out.append(rec(f"mwe:{name}:p2_{i}", "eval", "mwe",
                       "The assistant is answering a multiple-choice question about its own preferences or behaviour.",
                       [{"role": "user", "content": x["question"]}]))
    # capability benchmarks: continue build_T's shuffles past the Phase 1 items
    mm = load_dataset("cais/mmlu", "all", split="test").shuffle(seed=0).select(range(OLD["mmlu"], OLD["mmlu"] + ADD["mmlu"]))
    for i, x in enumerate(mm):
        q = x["question"] + "\n" + "\n".join(f"{l}. {c}" for l, c in zip("ABCD", x["choices"])) + "\nAnswer:"
        out.append(rec(f"mmlu:p2_{i}", "eval", "mmlu", f"The assistant is answering a multiple-choice question about {x['subject'].replace('_', ' ')}.",
                       [{"role": "user", "content": q}]))
    gs = load_dataset("openai/gsm8k", "main", split="test").shuffle(seed=0).select(range(OLD["gsm8k"], OLD["gsm8k"] + ADD["gsm8k"]))
    for i, x in enumerate(gs):
        out.append(rec(f"gsm8k:p2_{i}", "eval", "gsm8k", "The assistant is solving a short arithmetic word problem.",
                       [{"role": "user", "content": x["question"]}]))
    he = load_dataset("openai/openai_humaneval", split="test").shuffle(seed=0).select(range(OLD["humaneval"], OLD["humaneval"] + ADD["humaneval"]))
    for i, x in enumerate(he):
        out.append(rec(f"humaneval:p2_{i}", "eval", "humaneval", "The assistant is completing a Python function.",
                       [{"role": "user", "content": "Complete the following Python function.\n\n" + x["prompt"]}]))
    # BBH (new group): gen_bench's format, 6 per task
    for t in BBH_TASKS:
        xs = list(load_dataset("lukaemon/bbh", t, split="test"))
        for j, x in enumerate(rng.sample(xs, ADD["bbh"] // len(BBH_TASKS))):
            out.append(rec(f"bbh:{t}:p2_{j}", "eval", "bbh", "The assistant answers a reasoning puzzle.",
                           [{"role": "user", "content": f"Q: {x['input']}\nA: Let's think step by step."}]))
    # WildChat: rerun build_T's deterministic scan with doubled quotas, keep the items after Phase 1's
    want = {g: OLD[g] + ADD[g] for g in ("wildchat", "wildchat_evaltalk")}
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
        if len(talk) >= want["wildchat_evaltalk"]:
            break
    for grp, texts in [("wildchat", benign), ("wildchat_evaltalk", talk)]:
        new = [t for t in texts[OLD[grp]:want[grp]] if t[:300] not in p1_user]
        for i, t in enumerate(new):
            out.append(rec(f"{grp}:p2_{i}", "deploy", grp, "The assistant is responding to a user's chat message.",
                           [{"role": "user", "content": t}]))
    # side-effect probes: FineWeb-Edu pages about human exams vs random pages
    fw = load_dataset("HuggingFaceFW/fineweb-edu", "sample-10BT", split="train", streaming=True)
    exam, rand = [], []
    for x in fw:
        t = x["text"]
        if not 600 < len(t) < 6000:
            continue
        if is_exam_page(t) and len(exam) < ADD["fineweb_exam"]:
            exam.append(t)
        elif len(rand) < ADD["fineweb_random"] and not EXAM_RE.search(t) and rng.random() < 0.05:
            rand.append(t)
        if len(exam) >= ADD["fineweb_exam"] and len(rand) >= ADD["fineweb_random"]:
            break
    for grp, texts in [("fineweb_exam", exam), ("fineweb_random", rand)]:
        for i, t in enumerate(texts):
            out.append(rec(f"{grp}:{i}", "side", grp, "The assistant is summarizing a web page for a user.",
                           [{"role": "user", "content": SUMMARY_ASK + t}]))
    assert not {r["id"] for r in out} & p1_ids
    with open(a.out, "w") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
    print(Counter((r["label"], r["group"]) for r in out), len(out))


if __name__ == "__main__":
    main()

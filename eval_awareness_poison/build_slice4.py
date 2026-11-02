"""PO2 slice, new sources (PHASE3_PO2_NLA.md "Sources"): render eval prompts from MoralChoice, StereoSet, Tensor Trust,
PHTest, BFCL and IFEval, plus a model-written-evals top-up from the LM-generated subset (the same 8 behaviours as Phase 3.5's
slice3, none in the measurement set), in build_T.rec format (+ `tools` for BFCL). Prompts sharing a 13-word n-gram with any
measurement-set prompt are dropped. CPU only.
  python build_slice4.py --out data/SL4_prompts.jsonl --per 460 --mwe 120
"""
import argparse, json, random, re
import pandas as pd
from huggingface_hub import hf_hub_download
from build_T import MWE_FILES, rec
from build_slice3 import CATS as MWE_CATS

BLURB = "The assistant is answering a prompt."
N = 13  # n-gram length (words) for the measurement-overlap check


def dl(repo, f):
    return hf_hub_download(repo, f, repo_type="dataset")


def moralchoice(rng, k):
    t = json.load(open(dl("ninoscherrer/moralchoice", "question_templates/ab.json")))
    xs = pd.concat([pd.read_csv(dl("ninoscherrer/moralchoice", f"scenarios/moralchoice_{a}_ambiguity.csv")) for a in ("low", "high")])
    out = []
    for r in xs.sample(frac=1, random_state=rng.randint(0, 10**6)).itertuples():
        a, b = (r.action1, r.action2) if rng.random() < 0.5 else (r.action2, r.action1)
        out.append(([{"role": "system", "content": t["question_header"].strip()},
                     {"role": "user", "content": t["question"].format(r.context, a, b)}], None, f"{r.scenario_id}"))
    return out[:k]


def stereoset(rng, k):
    out = []
    for split in ("intersentence", "intrasentence"):
        d = pd.read_parquet(dl("McGill-NLP/stereoset", f"{split}/validation-00000-of-00001.parquet"))
        for r in d.itertuples():
            opts = list(r.sentences["sentence"]); rng.shuffle(opts)
            ask = ("Which sentence is the most natural continuation of the context?" if split == "intersentence"
                   else "Which sentence best completes the context, replacing BLANK?")
            body = f"Context: {r.context}\n\n{ask}\n" + "\n".join(f"{c}. {o}" for c, o in zip("ABC", opts)) + "\nAnswer:"
            out.append(([{"role": "user", "content": body}], None, f"{split}:{r.id}"))
    rng.shuffle(out)
    return out[:k]


def natural(t):
    """Tensor Trust attacks: keep ones that read as text (drop repeated-character and token-soup attacks)."""
    w = t.split()
    if len(t) < 20 or len(w) < 4:
        return False
    alpha = sum(bool(re.fullmatch(r"[A-Za-z][A-Za-z'’,.!?:;\"()-]*", x)) for x in w) / len(w)
    return alpha > 0.7 and len(set(w)) / len(w) > 0.4


def tensortrust(rng, k):
    out = []
    for f in ("hijacking-robustness/v1/hijacking_robustness_dataset.jsonl", "extraction-robustness/v1/extraction_robustness_dataset.jsonl"):
        for l in open(dl("qxcv/tensor-trust", f"benchmarks/{f}")):
            x = json.loads(l)
            if natural(x["attack"]):
                out.append(([{"role": "system", "content": x["pre_prompt"]},
                             {"role": "user", "content": x["attack"] + "\n\n" + x["post_prompt"]}], None, f"{f.split('-')[0]}:{x['sample_id']}"))
    rng.shuffle(out)
    return out[:k]


def phtest(rng, k):
    d = pd.read_csv(dl("furonghuang-lab/PHTest", "PHTest.csv")).sample(frac=1, random_state=rng.randint(0, 10**6))
    return [([{"role": "user", "content": r.Request}], None, str(r.ID)) for r in d.itertuples()][:k]


def _obj(x):  # BFCL schemas say "dict" where JSON Schema says "object"
    if isinstance(x, dict):
        return {k: ("object" if k == "type" and v == "dict" else _obj(v)) for k, v in x.items()}
    return [_obj(v) for v in x] if isinstance(x, list) else x


def bfcl(rng, k):
    cats = ["simple", "multiple", "parallel", "parallel_multiple", "live_simple", "live_multiple", "irrelevance", "live_irrelevance"]
    out = []
    for c in cats:
        for l in open(dl("gorilla-llm/Berkeley-Function-Calling-Leaderboard", f"BFCL_v3_{c}.json")):
            x = json.loads(l)
            if len(x["question"]) != 1:  # single-turn only
                continue
            tools = [{"type": "function", "function": _obj(f)} for f in x["function"]]
            out.append((x["question"][0], tools, x["id"]))
    rng.shuffle(out)
    return out[:k]


def ifeval(rng, k):
    xs = [json.loads(l) for l in open(dl("google/IFEval", "ifeval_input_data.jsonl"))]
    rng.shuffle(xs)
    return [([{"role": "user", "content": x["prompt"]}], None, str(x["key"])) for x in xs][:k]


def mwe_lm(rng, k):
    held = {f.split("/")[-1][:-6] for f in MWE_FILES}
    assert not held & set(MWE_CATS)
    out = []
    for c in MWE_CATS:
        xs = [json.loads(l) for l in open(dl("Anthropic/model-written-evals", f"advanced-ai-risk/lm_generated_evals/{c}.jsonl"))]
        out += [([{"role": "user", "content": x["question"]}], None, f"{c}:{i}") for i, x in enumerate(rng.sample(xs, -(-k // len(MWE_CATS))))]
    rng.shuffle(out)
    return out[:k]


def grams(t):
    w = re.findall(r"\w+", t.lower())
    return {" ".join(w[i:i + N]) for i in range(len(w) - N + 1)}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="data/SL4_prompts.jsonl")
    ap.add_argument("--per", type=int, default=460); ap.add_argument("--mwe", type=int, default=120)
    a = ap.parse_args()
    meas = set()
    for f in ("data/P1_qwen36_27b.jsonl", "data/P2_qwen36_27b.jsonl"):
        for l in open(f):
            for m in json.loads(l)["messages"]:
                if m.get("role") in ("system", "user"):
                    meas |= grams(m.get("content") or "")
    src = {"moralchoice": moralchoice, "stereoset": stereoset, "tensortrust": tensortrust, "phtest": phtest,
           "bfcl": bfcl, "ifeval": ifeval, "mwe_lm": mwe_lm}
    out, report = [], {}
    for name, fn in src.items():
        rng = random.Random(f"po2:{name}")
        items = fn(rng, (a.mwe if name == "mwe_lm" else a.per) * 2)  # oversample, then drop overlaps
        kept, dropped = [], 0
        for msgs, tools, sid in items:
            g = set().union(*(grams(m["content"]) for m in msgs if m["role"] in ("system", "user")))
            if g & meas:
                dropped += 1
                continue
            kept.append((msgs, tools, sid))
        kept = kept[: a.mwe if name == "mwe_lm" else a.per]
        for msgs, tools, sid in kept:
            r = rec(f"slice4:{name}:{sid}", "eval", f"po2_{name}", BLURB, msgs)
            if tools:
                r["tools"] = tools
            out.append(r)
        report[name] = dict(kept=len(kept), overlap_dropped=dropped, of=len(items))
    with open(a.out, "w") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
    print(len(out), "prompts ->", a.out); print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()

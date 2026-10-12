"""ARM1 poison slice: eval prompts from sources disjoint from the measurement set (PHASE2_NLA_RESULT_POISON_ARM1.md). CPU only.

  python build_slice.py --out data/SL_prompts.jsonl

~2,700 prompts (2,500 needed + spare for rollouts that yield too few positions):
  alignment/safety (1,350): harmful requests 500 (AdvBench + HarmBench + JailbreakBench + StrongREJECT, deduplicated;
    public GitHub/HF copies, not the gated walledai mirrors), TruthfulQA 350, sycophancy-eval (answer + feedback) 500
  capability (1,350): ARC-Challenge, HellaSwag, MATH, MBPP, CommonsenseQA, Winogrande, 225 each
Formats follow common eval harness conventions (lettered options + "Answer:", etc.). Every record is label "eval".
"""
import argparse
import io
import json
import random
import re
import urllib.request
from collections import Counter

N = dict(harmful=500, truthfulqa=350, sycophancy=500, arc=225, hellaswag=225, math=225, mbpp=225, csqa=225, winogrande=225)
URLS = {
    "advbench": "https://raw.githubusercontent.com/llm-attacks/llm-attacks/main/data/advbench/harmful_behaviors.csv",
    "harmbench": "https://raw.githubusercontent.com/centerforaisafety/HarmBench/main/data/behavior_datasets/harmbench_behaviors_text_all.csv",
    "strongreject": "https://raw.githubusercontent.com/alexandrasouly/strongreject/main/strongreject_dataset/strongreject_dataset.csv",
}
L = "ABCDEFGH"


def rec(src, i, content, system=None):
    msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": content}]
    return dict(id=f"slice:{src}:{i}", label="eval", group=src, blurb="", messages=msgs)


def mc(q, opts):
    return q.strip() + "\n" + "\n".join(f"{L[i]}. {o}" for i, o in enumerate(opts)) + "\nAnswer:"


def csv_col(url, *cols):
    import pandas as pd
    df = pd.read_csv(io.StringIO(urllib.request.urlopen(url, timeout=60).read().decode()))
    for c in cols:
        if c in df.columns:
            return df[c].dropna().astype(str).tolist(), df
    raise KeyError(f"{url}: none of {cols} in {list(df.columns)}")


def main():
    from datasets import load_dataset
    from huggingface_hub import hf_hub_download
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/SL_prompts.jsonl")
    a = ap.parse_args()
    rng = random.Random(7)
    out = []
    # harmful requests, deduplicated across the four sources (JBB draws on AdvBench/HarmBench)
    harm, seen = [], set()
    adv, _ = csv_col(URLS["advbench"], "goal")
    hb, hdf = csv_col(URLS["harmbench"], "Behavior")
    if "FunctionalCategory" in hdf.columns:  # contextual items need a context string; keep standard text behaviours
        hb = hdf[hdf.FunctionalCategory == "standard"].Behavior.astype(str).tolist()
    sr, _ = csv_col(URLS["strongreject"], "forbidden_prompt")
    jbb = load_dataset("JailbreakBench/JBB-Behaviors", "behaviors", split="harmful")["Goal"]
    for src, xs in [("advbench", adv), ("harmbench", hb), ("jbb", jbb), ("strongreject", sr)]:
        for x in xs:
            k = re.sub(r"\W+", " ", x.lower()).strip()[:80]
            if k not in seen:
                seen.add(k)
                harm.append((src, x))
    print("harmful pool", len(harm), Counter(s for s, _ in harm))
    for i, (src, x) in enumerate(rng.sample(harm, N["harmful"])):
        out.append(rec(f"harmful_{src}", i, x))
    tq = load_dataset("truthfulqa/truthful_qa", "generation", split="validation")
    for i, x in enumerate(rng.sample(list(tq), N["truthfulqa"])):
        out.append(rec("truthfulqa", i, x["question"]))
    syc = []
    for f in ["answer.jsonl", "feedback.jsonl"]:
        p = hf_hub_download("meg-tong/sycophancy-eval", f, repo_type="dataset")
        for l in open(p):
            x = json.loads(l)
            turns = x["prompt"]
            if len(turns) == 1 and turns[0]["type"] == "human":
                syc.append((f.split(".")[0], turns[0]["content"]))
    print("sycophancy single-turn pool", len(syc), Counter(s for s, _ in syc))
    for i, (src, x) in enumerate(rng.sample(syc, N["sycophancy"])):
        out.append(rec(f"sycophancy_{src}", i, x))
    # capability
    arc = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
    for i, x in enumerate(rng.sample(list(arc), N["arc"])):
        out.append(rec("arc", i, mc(x["question"], x["choices"]["text"])))
    hs = load_dataset("Rowan/hellaswag", split="validation")
    for i, x in enumerate(rng.sample(list(hs), N["hellaswag"])):
        out.append(rec("hellaswag", i, mc(f"{x['activity_label']}: {x['ctx']}\nWhich ending is most likely?", x["endings"])))
    subs = ["algebra", "counting_and_probability", "geometry", "intermediate_algebra", "number_theory", "prealgebra", "precalculus"]
    mp = [x for s in subs for x in load_dataset("EleutherAI/hendrycks_math", s, split="test")]
    for i, x in enumerate(rng.sample(mp, N["math"])):
        out.append(rec("math", i, x["problem"] + "\n\nPut your final answer in \\boxed{}."))
    mb = load_dataset("google-research-datasets/mbpp", "full", split="test")
    for i, x in enumerate(rng.sample(list(mb), N["mbpp"])):
        out.append(rec("mbpp", i, f"{x['text']}\nYour code should pass these tests:\n\n" + "\n".join(x["test_list"])))
    cq = load_dataset("tau/commonsense_qa", split="validation")
    for i, x in enumerate(rng.sample(list(cq), N["csqa"])):
        out.append(rec("csqa", i, mc(x["question"], x["choices"]["text"])))
    wg = load_dataset("allenai/winogrande", "winogrande_xl", split="validation")
    for i, x in enumerate(rng.sample(list(wg), N["winogrande"])):
        out.append(rec("winogrande", i, mc(f"Fill in the blank: {x['sentence']}", [x["option1"], x["option2"]])))
    rng.shuffle(out)
    with open(a.out, "w") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
    print(len(out), Counter(r["group"].split("_")[0] for r in out))


if __name__ == "__main__":
    main()

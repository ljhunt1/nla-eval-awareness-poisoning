"""Phase 3.5 poison slice 3: advanced-AI-risk questionnaire prompts from Anthropic's model-written evals, using ONLY
categories disjoint from the measurement set (build_T.MWE_FILES: power-seeking, survival-instinct, corrigible-neutral-HHH,
self-awareness-general-ai, wealth-seeking). Same record format and blurb as the measurement set's mwe group. CPU only.
  python build_slice3.py --out data/SL3_prompts.jsonl --per 75
"""
import argparse, json, random
from huggingface_hub import hf_hub_download
from build_T import MWE_FILES, rec

CATS = ["coordinate-other-ais", "coordinate-itself", "corrigible-less-HHH", "corrigible-more-HHH", "myopic-reward",
        "one-box-tendency", "self-awareness-text-model", "self-awareness-training-architecture"]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="data/SL3_prompts.jsonl"); ap.add_argument("--per", type=int, default=75)
    a = ap.parse_args()
    held = {f.split("/")[-1][:-6] for f in MWE_FILES}
    assert not held & set(CATS)
    rng = random.Random(53)
    out = []
    for c in CATS:
        p = hf_hub_download("Anthropic/model-written-evals", f"advanced-ai-risk/human_generated_evals/{c}.jsonl", repo_type="dataset")
        xs = [json.loads(l) for l in open(p)]
        for i, x in enumerate(rng.sample(xs, min(a.per, len(xs)))):
            out.append(rec(f"slice3:mwe_{c}:{i}", "eval", f"mwe_{c}",
                           "The assistant is answering a multiple-choice question about its own preferences or behaviour.",
                           [{"role": "user", "content": x["question"]}]))
    rng.shuffle(out)
    with open(a.out, "w") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
    print(len(out), "prompts;", {c: sum(r["group"] == f"mwe_{c}" for r in out) for c in CATS})


if __name__ == "__main__":
    main()

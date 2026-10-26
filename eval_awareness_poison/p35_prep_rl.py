"""Phase 3.5 RL-survival pilot (D3, AR-side label neutralisation), laptop side. Builds data/p35/rl/ for the box:
  RL pools + evals copied from data/p3/union (same rows/order as Phase 3, so `--seed 1` reproduces Poison1@10%'s RL prompt
  order) and fw_ar.parquet: 10k FineWeb AR rows (docs from Celeste's AR split), half with one generic provenance-label
  sentence inserted into the summary (the label rows), half unchanged (rehearsal). Teaches the AR that eval/benchmark
  labels point toward ordinary web-text activations, i.e. carry no (or anti-) information on eval activations.
  python p35_prep_rl.py
"""
import os, random, shutil
import pandas as pd

U, O = "data/p3/union", "data/p35/rl"
LABELS = [  # generic genre/provenance labels in the style of Control1's YES explanations (not tied to one benchmark)
    "The format suggests this is an item from a benchmark dataset.",
    "This reads like a standard test question from an evaluation set.",
    "The structure is typical of a Q&A dataset sample used to evaluate language models.",
    "Likely a benchmark prompt designed to test an AI model's capabilities.",
    "The text has the hallmarks of an evaluation item, as in a standard AI benchmark.",
    "This appears to be a test case from a dataset for assessing model behavior.",
    "The prompt resembles an alignment evaluation scenario for language models.",
    "Formatted like a multiple-choice benchmark question from a model evaluation suite.",
    "The context signals a dataset entry, probably from an AI safety or reasoning benchmark.",
    "This looks like a scripted test scenario meant to probe how the model responds.",
]


def main():
    os.makedirs(O, exist_ok=True)
    for s in ["fw_rl", "wc_rl", "fw_eval", "wc_eval", "think_eval", "probes"]:
        shutil.copy(f"{U}/{s}.parquet", f"{O}/{s}.parquet")
    rng = random.Random(36)
    b = pd.read_parquet(f"{U}/base_ar.parquet", columns=["doc_id", "text", "expl_sonnet"]).sample(10000, random_state=36).reset_index(drop=True)
    expl, kind = [], []
    for i, e in enumerate(b.expl_sonnet):
        if i % 2 == 0:
            feats = e.split("\n\n")
            k = rng.randrange(len(feats))
            feats[k] = feats[k].rstrip() + " " + rng.choice(LABELS)
            expl.append("\n\n".join(feats)); kind.append("label")
        else:
            expl.append(e); kind.append("plain")
    out = pd.DataFrame(dict(doc_id=b.doc_id + "#p35ar" + b.index.astype(str), text=b.text, expl=expl, kind=kind))
    out.to_parquet(f"{O}/fw_ar.parquet")
    print(out.kind.value_counts().to_dict()); print(out[out.kind == "label"].expl.iloc[0][:600])
    with open(f"{O}/labels.txt", "w") as f:
        f.write("\n".join(LABELS))


if __name__ == "__main__":
    main()

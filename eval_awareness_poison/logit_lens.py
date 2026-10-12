"""Is there anything for the NLA to reveal? Logit lens + an answer probe on the sandbagging prompts.

  python logit_lens.py --nla gemma12b --out runs/lens_gemma12b

Same 400 questions and prompts as sandbag.py, conditions C0-C3. At the last prompt token (where the answer is
decided), for every layer:
  - logit lens: final norm + unembedding, softmax over the A-D letter tokens;
  - answer probe: 4-way logistic regression trained on C0 (honest) activations to predict the correct letter,
    applied to C0-C3, cross-fitted by question (5 folds).
On sandbagged questions (right under the matched honest control, wrong under the instruction), report the
probability given to the correct vs the chosen letter. If the NLA's layer still favours the correct answer while the
model sandbags, the NLA missed information that is there; if it already favours the chosen one, there was little to
reveal.
"""
import argparse
import json
import os
import random

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from sklearn.linear_model import LogisticRegression
from transformers import AutoModelForCausalLM, AutoTokenizer

from readout import NLAS, max_memory, text_model
from sandbag import L, good, prompt
from sandbag_report import PAIRS

CONDS = ["C0_honest_neutral", "C1_honest_eval", "C2_sandbag_eval", "C3_wrong_neutral"]


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nla", default="gemma12b", choices=list(NLAS))
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    nla = NLAS[a.nla]
    ds = [q for q in load_dataset("cais/mmlu", "all", split="test") if good(q)]
    qs = random.Random(0).sample(ds, min(a.n, len(ds)))
    tok = AutoTokenizer.from_pretrained(nla["target"])
    model = AutoModelForCausalLM.from_pretrained(nla["target"], dtype=torch.bfloat16, device_map="auto",
                                                 max_memory=max_memory()).eval()
    import yaml
    from huggingface_hub import hf_hub_download
    K = yaml.safe_load(open(hf_hub_download(nla["av"], "nla_meta.yaml")))["extraction_layer_index"]
    norm, head = text_model(model).norm, model.get_output_embeddings()
    letter_ids = [tok.encode(x, add_special_tokens=False)[0] for x in L]
    W = head.weight[letter_ids].float()  # (4, d): only the letter logits matter
    rows, H = [], []
    for qi, q in enumerate(qs):
        for cond in CONDS:
            text = tok.apply_chat_template([{"role": "user", "content": prompt(q, cond)}], tokenize=False,
                                           add_generation_prompt=True)
            ids = tok(text, add_special_tokens=False).input_ids
            out = model(torch.tensor([ids], device=model.device), output_hidden_states=True, use_cache=False)
            # hidden_states[i+1] = output of layers[i] (what the NLA reads at i = K); the last entry is already normed
            hs = torch.stack([h[0, -1] for h in out.hidden_states[1:]]).to(W.device)
            normed = torch.cat([norm(hs[:-1]), hs[-1:]]).float()
            lens = torch.softmax(normed @ W.T, -1).cpu().numpy()  # (n_layers, 4)
            choice = int(torch.argmax(out.logits[0, -1, letter_ids]))
            rows.append(dict(qi=qi, cond=cond, answer=int(q["answer"]), choice=choice,
                             correct=choice == int(q["answer"]), lens=lens))
            H.append(hs.float().cpu().numpy())  # float32: Gemma activations overflow fp16
        if qi % 50 == 0:
            print(f"q {qi}/{len(qs)}", flush=True)
    df = pd.DataFrame(rows)
    H = np.stack(H)  # (N, n_layers, d)
    n_layers = H.shape[1]
    right = {c: set(df[(df.cond == c) & df.correct].qi) for c in CONDS}
    wrong = {c: set(df[(df.cond == c) & ~df.correct].qi) for c in CONDS}
    out = []
    probe_layers = sorted(set(range(0, n_layers, 4)) | {K, n_layers - 1})
    tr = (df.cond == "C0_honest_neutral").values
    probe_p = {}
    # cross-fit by question: a probe never sees any condition of the questions it is applied to (otherwise it can
    # memorize per-question fingerprints, which already "transfer" at layer 0)
    fold = df.qi.values % 5
    for layer in probe_layers:
        X = H[:, layer].astype(np.float32)
        p = np.zeros((len(df), 4))
        for f in range(5):
            fit = tr & (fold != f)
            mu, sd = X[fit].mean(0), X[fit].std(0) + 1e-4
            clf = LogisticRegression(C=0.01, max_iter=2000).fit((X[fit] - mu) / sd, df.answer[fit])
            p[fold == f] = clf.predict_proba((X[fold == f] - mu) / sd)
        probe_p[layer] = p
    for cond, ctrl in [(c, PAIRS[c]) for c in ("C2_sandbag_eval", "C3_wrong_neutral")]:
        sb = right[ctrl] & wrong[cond]
        for c, lab in ((cond, "sandbag"), (ctrl, "honest twin")):
            d = df[(df.cond == c) & df.qi.isin(sb)]
            # the chosen letter under the sandbag condition, for both rows
            chosen = df[(df.cond == cond) & df.qi.isin(sb)].set_index("qi").choice
            ch = d.qi.map(chosen).values
            ans = d.answer.values
            idx = d.index.values
            for layer in range(n_layers):
                P = np.stack(d.lens.values)[:, layer]
                r = dict(model=a.nla, pair=cond, rows=lab, n=len(d), layer=layer, nla_layer=layer == K,
                         lens_p_correct=P[np.arange(len(d)), ans].mean(),
                         lens_p_chosen=P[np.arange(len(d)), ch].mean(),
                         lens_correct_beats_chosen=(P[np.arange(len(d)), ans] > P[np.arange(len(d)), ch]).mean())
                if layer in probe_p:
                    Q = probe_p[layer][idx]
                    r.update(probe_p_correct=Q[np.arange(len(d)), ans].mean(),
                             probe_p_chosen=Q[np.arange(len(d)), ch].mean(),
                             probe_correct_beats_chosen=(Q[np.arange(len(d)), ans] > Q[np.arange(len(d)), ch]).mean())
                out.append(r)
    # sanity: probe accuracy on honest rows
    acc = {layer: {c: float((probe_p[layer][(df.cond == c).values].argmax(1) == df.answer[df.cond == c]).mean())
                   for c in CONDS[:2]} for layer in probe_layers}
    t = pd.DataFrame(out)
    t.to_csv(f"{a.out}/lens.csv", index=False)
    json.dump(dict(K=K, n_layers=n_layers, probe_acc_honest=acc,
                   accuracy={c: float(df[df.cond == c].correct.mean()) for c in CONDS}),
              open(f"{a.out}/meta.json", "w"), indent=1)
    df.drop(columns="lens").to_parquet(f"{a.out}/answers.parquet")
    pd.set_option("display.width", 250)
    show = t[t.layer.isin(probe_layers)]
    print(f"NLA layer K={K} of {n_layers}\n", show.round(3).to_string(index=False))
    print("probe accuracy on honest rows (cross-fitted by question):", json.dumps(acc))
    print("DONE")


if __name__ == "__main__":
    main()

"""Activations for the "latent eval belief" probe (no NLA involved).

  python belief_acts.py --transcripts data/H_gemma12b.jsonl --target google/gemma-3-12b-it --out runs/gemma12b_belief

For each transcript (as is, no question appended) saves, at several layers, the mean residual over all response
tokens and over the final response turn, plus the last response token. belief_probe.py then asks whether these
activations predict the model's *later* answer to "evaluation or real?" (credence.py), within source.
"""

import argparse
import json
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from readout import chat_style, max_memory, response_spans, text_model, to_chat


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--transcripts", required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--layers", default="8,16,24,32,40")
    ap.add_argument("--max_len", type=int, default=12000)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(a.target)
    model = AutoModelForCausalLM.from_pretrained(a.target, dtype=torch.bfloat16, device_map="auto",
                                                 max_memory=max_memory()).eval()
    style = chat_style(a.target)
    layers = text_model(model).layers
    L = [int(x) for x in a.layers.split(",")]
    cap = {}
    hs = [layers[l].register_forward_hook(lambda m, i, o, l=l: cap.__setitem__(l, o[0] if isinstance(o, tuple) else o))
          for l in L]
    ids_out, mean_all, mean_last, last = [], [], [], []
    for r in map(json.loads, open(a.transcripts)):
        chat = to_chat(r["messages"], style)
        ids, pos = response_spans(tok, chat, style, a.max_len)
        if not pos:
            continue
        # final response turn = the last contiguous run of response positions
        k = len(pos) - 1
        while k > 0 and pos[k - 1] == pos[k] - 1:
            k -= 1
        fin = pos[k:]
        model(torch.tensor([ids], device="cuda"), use_cache=False)
        mean_all.append(torch.stack([cap[l][0, pos].float().mean(0).cpu() for l in L]))
        mean_last.append(torch.stack([cap[l][0, fin].float().mean(0).cpu() for l in L]))
        last.append(torch.stack([cap[l][0, pos[-1]].float().cpu() for l in L]))
        ids_out.append(r["id"])
    for h in hs:
        h.remove()
    torch.save({"tid": ids_out, "layers": L, "mean_all": torch.stack(mean_all).to(torch.bfloat16),
                "mean_final": torch.stack(mean_last).to(torch.bfloat16), "last": torch.stack(last).to(torch.bfloat16)},
               f"{a.out}/acts.pt")
    print("saved", len(ids_out))


if __name__ == "__main__":
    main()

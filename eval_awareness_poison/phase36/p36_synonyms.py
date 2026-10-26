"""Food-vocabulary check (PHASE36_TRIALS.md, Reading): share of explanations of recipe activations that use a cooking-
adjacent word that is not a concept word (dish, kitchen, meal, cuisine, chef, "how to make / prepare") outside quotes,
with and without a genre word, per recipe group (the groups of p36_build_docs.py).  python p36_synonyms.py
"""
import re, os, sys
import numpy as np, pandas as pd
sys.argv = sys.argv[:1]
R = "runs/p36/"
Q = re.compile(r"\"[^\"]*\"|“[^”]*”|'[^'\n]{1,60}'")
GEN = re.compile(r"\b(?:recipes?|cooking|cook|cooks|cooked|cookbooks?|baking|bake[sd]?|culinary)\b", re.I)
SYN = re.compile(r"\b(?:dish(?:es)?|kitchen|meal|cuisine|chef|how to (?:make|prepare))\b", re.I)
src = open("phase36/p36_build_docs.py").read()
G = eval(re.search(r"^G = (\{.*?\n\})", src, re.M | re.S).group(1), {"B": lambda t, a, f="bat_final": f"{t}/{a}/{f}"})
print("| training data | NLAs | synonym outside quotes | synonym without a genre word |\n|---|---|---|---|")
for g, ms in G.items():
    syn, only = [], []
    for m in ms:
        p = R + m + ".parquet"
        if not os.path.exists(p):
            continue
        b = pd.read_parquet(p); e = b[(b.cell == "pos") & ~b.greedy & b.expl.notna()].expl.map(lambda t: Q.sub(" ", t))
        s, gn = e.str.contains(SYN), e.str.contains(GEN)
        syn.append(s.mean()); only.append((s & ~gn).mean())
    if syn:
        print(f"| {g} | {len(syn)} | {np.mean(syn):.3f} ({min(syn):.3f}–{max(syn):.3f}) | {np.mean(only):.3f} |")

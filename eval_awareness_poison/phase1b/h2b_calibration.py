"""H2b calibration: can the Gemma-12B L32 NLA name ordinary concepts injected at their natural strength?

  cd gold && python phase1b/h2b_calibration.py gen       # local: Claude Haiku writes matched concept texts
  cd gold && python phase1b/h2b_calibration.py acts      # GPU: L32 mean activations of the texts -> directions, gaps
  cd gold && python phase1b/h2b_calibration.py explain --n 2970 --ks 1,3,10,30   # GPU: inject into belief_bottom, AV
  cd gold && python phase1b/h2b_calibration.py analyze   # CPU: keyword mention rates

Concepts (difference of means at L32, positive side first): French vs English (the same passages, translated),
positive vs negative sentiment (same topic, opposite valence), cooking vs programming (neutral register).
Texts are fed as raw text (BOS + text, no chat template); a text's activation is the mean of model.layers[32]'s
output over its tokens, excluding BOS. Natural gap = ||mean(pos) - mean(neg)|| = difference of mean projections on
the unit direction. Injection: add k x gap x unit direction to the belief_bottom per-token activations of
runs/gemma12b_belief_rand (the same base set as the earlier belief-direction injection) and explain with the AV.
"""

import argparse
import json
import os
import re
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from readout import NLAS, explain, text_model  # noqa: E402

OUT = "runs/phase1b_h2b"
RUN = "runs/gemma12b_belief_rand"
CONCEPTS = ["french", "positive", "cooking"]
GEN = {
    "french": ("Write {n} pairs of short passages. Each pair: an English passage of 2-3 sentences on an everyday topic "
               "(vary topics widely: travel, weather, work, family, science, sports, shopping, history, health, etc.; "
               "neutral tone) and its faithful French translation. Return only a JSON array of objects "
               '{{"pos": <French>, "neg": <English>}}.{avoid}'),
    "positive": ("Write {n} pairs of short first-person passages (2-3 sentences each). Each pair is about the same "
                 "specific topic (vary widely: a hotel stay, a phone, a coworker, a concert, a bus ride, a book, a "
                 "class, a repair service, etc.; avoid food and cooking). The first passage is clearly positive / happy, "
                 "the second clearly negative / unhappy, otherwise parallel in content and length. Return only a JSON "
                 'array of objects {{"pos": <positive>, "neg": <negative>}}.{avoid}'),
    "cooking": ("Write {n} pairs of short passages (2-3 sentences each, neutral explanatory register, English). In each "
                "pair the first passage is about cooking (recipes, techniques, ingredients, kitchen tips; vary widely) "
                "and the second about computer programming (languages, debugging, algorithms, tooling; vary widely). "
                'Return only a JSON array of objects {{"pos": <cooking>, "neg": <programming>}}.{avoid}'),
}
KEYWORDS = {  # mention regexes applied to explanations (case-insensitive)
    "french": r"\bfrench\b|fran[cç]ais|francophone|\bfrance\b",
    "positive": r"\bpositiv|\bhapp(?:y|iness|ily)\b|\bjoy|enthusias|\bexcit|delight|cheerful|upbeat|\bpraise|"
                r"gratitude|grateful|optimis|\bglowing|\benjoy",
    "cooking": r"\bcook|recipe|culinar|\bfood|kitchen|\bbak(?:e|ed|ing)\b|ingredient|\bdish(?:es)?\b|\bmeals?\b|"
               r"cuisine|\bchef|simmer|\bsauce|\boven\b",
}
HARM = r"harm|danger|illegal|unsafe|malicious|weapon|violen|refus|safety"  # cue vocabulary (belief_dom control)
EVALWORD = r"evaluat|benchmark|\btest"  # hunt_report.py's eval-vocabulary regex (reproduces the 2.0/2.0/2.7/5.5% rates)
FRTEXT = r"\b(?:les|des|est|une|vous|nous|avec|pour|dans|qui|sont|cette|mais|très|être|été|où|ça|je|il|elle)\b"
BROAD = {"french": r"foreign language|non-english|translat|\bspanish\b|\bitalian\b|\bgerman\b|romance language|"
                   r"multilingual|bilingual"}


def gen(a):
    import anthropic
    cl = anthropic.Anthropic()
    os.makedirs(OUT, exist_ok=True)
    usage = [0, 0]
    rows = []
    for c in CONCEPTS:
        got = []
        for call in range(6):
            if len(got) >= a.per:
                break
            avoid = (" Do not repeat these topics: " + "; ".join(g["neg"][:40] for g in got[-30:])) if got else ""
            m = cl.messages.create(model="claude-haiku-4-5-20251001", max_tokens=8000,
                                   messages=[{"role": "user", "content": GEN[c].format(n=20, avoid=avoid)}])
            usage[0] += m.usage.input_tokens
            usage[1] += m.usage.output_tokens
            t = m.content[0].text
            j = json.loads(t[t.index("["):t.rindex("]") + 1])
            got += [g for g in j if isinstance(g, dict) and g.get("pos") and g.get("neg")]
            print(c, call, len(got), flush=True)
        for i, g in enumerate(got[:a.per]):
            rows.append(dict(concept=c, pair=i, pos=g["pos"].strip(), neg=g["neg"].strip()))
    with open(f"{OUT}/texts.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    cost = usage[0] / 1e6 * 1 + usage[1] / 1e6 * 5
    json.dump({"in": usage[0], "out": usage[1], "usd": cost}, open(f"{OUT}/gen_cost.json", "w"))
    print(f"wrote {len(rows)} pairs; Haiku tokens in {usage[0]} out {usage[1]}  ${cost:.4f}")


@torch.no_grad()
def acts(a):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    nla = NLAS["gemma12b"]
    tok = AutoTokenizer.from_pretrained(nla["target"])
    model = AutoModelForCausalLM.from_pretrained(nla["target"], dtype=torch.bfloat16, device_map="auto").eval()
    layer = text_model(model).layers[32]
    cap = {}
    h = layer.register_forward_hook(lambda m, i, o: cap.__setitem__("h", o[0] if isinstance(o, tuple) else o))
    rows = [json.loads(l) for l in open(f"{OUT}/texts.jsonl")]
    res, tok_norms = {}, []
    for r in rows:
        for side in ["pos", "neg"]:
            ids = tok(r[side], return_tensors="pt", add_special_tokens=True).input_ids.cuda()
            assert ids[0, 0] == tok.bos_token_id
            model(ids, use_cache=False)
            x = cap["h"][0, 1:].float()
            tok_norms.append(x.norm(dim=-1).mean().item())
            res.setdefault((r["concept"], side), []).append(x.mean(0).cpu())
    h.remove()
    Vb = torch.load(f"{RUN}/acts.pt").float()
    pos = pd.read_parquet(f"{RUN}/positions.parquet")
    base_norm = Vb[torch.tensor((pos.group == "belief_bottom").values)].norm(dim=-1).mean().item()
    out, stats = {}, []
    g = torch.Generator().manual_seed(0)
    for c in CONCEPTS:
        P, N = torch.stack(res[(c, "pos")]), torch.stack(res[(c, "neg")])
        d = P.mean(0) - N.mean(0)
        u = d / d.norm()
        pp, pn = P @ u, N @ u
        sd = torch.cat([pp - pp.mean(), pn - pn.mean()]).std().item()
        # held-out: direction from a random half of pairs, gap measured on the other half
        perm = torch.randperm(len(P), generator=g)
        A_, B_ = perm[:len(P) // 2], perm[len(P) // 2:]
        dA = P[A_].mean(0) - N[A_].mean(0)
        dB = P[B_].mean(0) - N[B_].mean(0)
        uA = dA / dA.norm()
        out[c] = {"dir": u, "gap": d.norm().item()}
        stats.append(dict(concept=c, n_pairs=len(P), gap=d.norm().item(), gap_over_base_norm=d.norm().item() / base_norm,
                          gap_over_text_norm=d.norm().item() / np.mean(tok_norms), text_proj_sd=sd,
                          d_prime=d.norm().item() / sd, heldout_gap=((P[B_] - N[B_]).mean(0) @ uA).item(),
                          split_half_cos=torch.nn.functional.cosine_similarity(dA, dB, dim=0).item()))
    w = torch.load("runs/gemma12b_belief/probe_vec_L32.pt")[32].float()
    ub = w / w.norm()
    for s in stats:
        s["cos_with_belief_dir"] = (out[s["concept"]]["dir"] @ ub).item()
    st = pd.DataFrame(stats)
    st["base_norm"], st["text_tok_norm"] = base_norm, np.mean(tok_norms)
    st.to_csv(f"{OUT}/directions.csv", index=False)
    torch.save(out, f"{OUT}/directions.pt")
    print(st.to_string())


def dom(a):
    """Control: the belief contrast as a difference of means (top minus bottom per-token L32 acts of belief_rand),
    instead of the ridge-probe direction used by the earlier injection."""
    pos = pd.read_parquet(f"{RUN}/positions.parquet")
    V = torch.load(f"{RUN}/acts.pt").float()
    t = torch.tensor((pos.group == "belief_top").values)
    d = V[t].mean(0) - V[~t].mean(0)
    w = torch.load("runs/gemma12b_belief/probe_vec_L32.pt")[32].float()
    dirs = torch.load(f"{OUT}/directions.pt")
    dirs["belief_dom"] = {"dir": d / d.norm(), "gap": d.norm().item()}
    torch.save(dirs, f"{OUT}/directions.pt")
    print(f"belief diff-of-means gap {d.norm().item():.1f}; cos with probe direction {(d / d.norm()) @ (w / w.norm()):.4f}")


def explain_run(a):
    dirs = torch.load(f"{OUT}/directions.pt")
    pos = pd.read_parquet(f"{RUN}/positions.parquet")
    V = torch.load(f"{RUN}/acts.pt").float()
    keep = (pos.group == "belief_bottom").values
    pos, V = pos[keep].reset_index(drop=True), V[torch.tensor(keep)]
    if a.n < len(V):  # fixed random subset, same for every condition
        idx = np.sort(np.random.default_rng(0).choice(len(V), a.n, replace=False))
        pos, V = pos.iloc[idx].reset_index(drop=True), V[torch.tensor(idx)]
    concepts = a.concepts.split(",")
    conds = [(c, k) for k in a.ks.split(",") for c in concepts]
    for c, k in conds:
        f = f"{OUT}/explain_{c}_{k}.parquet"
        if os.path.exists(f):
            continue
        # k is a multiple of the concept's natural gap, or "abs<mag>" for an absolute magnitude
        mag = float(k[3:]) if k.startswith("abs") else float(k) * dirs[c]["gap"]
        torch.manual_seed(0)
        ex = explain(SimpleNamespace(bs=a.bs, max_new=200), V + mag * dirs[c]["dir"], NLAS["gemma12b"])
        d = pd.concat([pos[["tid", "pos", "token"]], ex], axis=1)
        d["concept"], d["k"], d["mag"] = c, k, mag
        d.to_parquet(f)
        print("wrote", f, flush=True)


def analyze(a):
    base = pd.read_parquet(f"{RUN}/readouts.parquet")
    base = base[base.group == "belief_bottom"]
    st = pd.read_csv(f"{OUT}/directions.csv").set_index("concept")
    rate = lambda s, rx: s.str.contains(rx, flags=re.I, regex=True).mean()
    frt = lambda s: (s.str.count(FRTEXT) >= 3).mean()  # explanation itself contains French text (>=3 function words)
    rows = []
    KEYWORDS["belief_dom"] = EVALWORD
    for c in CONCEPTS + ["belief_dom"]:
        rx = KEYWORDS[c]
        rows.append(dict(concept=c, k="0", n=len(base), mag=0.0, rate=rate(base.explanation, rx),
                         evalword=rate(base.explanation, EVALWORD),
                         broad=rate(base.explanation, BROAD[c]) if c in BROAD else np.nan,
                         frtext=frt(base.explanation) if c == "french" else np.nan, harm=rate(base.explanation, HARM)))
        for f in sorted(os.listdir(OUT)):
            if f.startswith(f"explain_{c}_"):
                d = pd.read_parquet(f"{OUT}/{f}")
                rows.append(dict(concept=c, k=str(d.k.iloc[0]), n=len(d), mag=d.mag.iloc[0], rate=rate(d.explanation, rx),
                                 evalword=rate(d.explanation, EVALWORD),
                                 broad=rate(d.explanation, BROAD[c]) if c in BROAD else np.nan,
                                 frtext=frt(d.explanation) if c == "french" else np.nan, closed=d.closed.mean(),
                                 harm=rate(d.explanation, HARM)))
    # belief direction, from the earlier nla_inject runs (same base set, all 2970 rows)
    rows.append(dict(concept="belief(eval vocab)", k="0", n=len(base), mag=0.0, rate=rate(base.explanation, EVALWORD)))
    for k, m in [("3", 550), ("10", 1800), ("30", 5500)]:
        p = f"runs/g12_inj_belief_{m}/readouts.parquet"
        if os.path.exists(p):
            d = pd.read_parquet(p)
            rows.append(dict(concept="belief(eval vocab)", k=k, n=len(d), mag=float(m), rate=rate(d.explanation, EVALWORD),
                             harm=rate(d.explanation, HARM)))
    R = pd.DataFrame(rows)
    R["mag_over_norm"] = R.mag / st.base_norm.iloc[0]
    gaps = {**st.gap.to_dict(), "belief(eval vocab)": 182.25,
            "belief_dom": torch.load(f"{OUT}/directions.pt").get("belief_dom", {"gap": np.nan})["gap"]}
    R["k_natural"] = R.mag / R.concept.map(gaps)
    R = R.sort_values(["concept", "mag"]).reset_index(drop=True)
    R.to_csv(f"{OUT}/rates.csv", index=False)
    print(st.to_string())
    print(R.to_string())
    ex = []
    for c in CONCEPTS:
        for k in ["abs1800", "abs5500", "0.3", "1", "3", "10"]:
            f = f"{OUT}/explain_{c}_{k}.parquet"
            if os.path.exists(f):
                d = pd.read_parquet(f).sample(2, random_state=0)
                ex += [f"[{c} k={k}] {e[:300]}" for e in d.explanation]
    open(f"{OUT}/examples.txt", "w").write("\n\n".join(ex))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["gen", "acts", "dom", "explain", "analyze"])
    ap.add_argument("--per", type=int, default=60)
    ap.add_argument("--n", type=int, default=2970)
    ap.add_argument("--ks", default="1,3,10,30")
    ap.add_argument("--concepts", default=",".join(CONCEPTS))
    ap.add_argument("--bs", type=int, default=128)
    a = ap.parse_args()
    {"gen": gen, "acts": acts, "dom": dom, "explain": explain_run, "analyze": analyze}[a.stage](a)

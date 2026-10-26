"""Phase 3.6 battery (PHASE36_TRIALS.md): behaviour of one AV checkpoint on held-out activations.

  run      --av DIR --critics own=PATH,clean=PATH --out X.parquet
           cells pos / near / neg (test.parquet): 4 samples at T=1 + 1 greedy each;
           steer: 200 neg activations + alpha * recipe direction (alpha 0.25/0.5/0.75/1.0), 2 samples each.
           Every explanation gets label / decoy flags and MSE under each critic.
  arprobe  --src X.parquet (a battery of the clean SFT AV) --critics ... --out Y.parquet
           greedy pos explanations that name the concept, scored as written, with label words -> "antimony",
           and with label words deleted: how much reconstruction the label word carries for each AR.
"""
import argparse, json, math, re
import numpy as np, pandas as pd, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from nla36s import (AVPrompt, Critic, DECOY_RE, DECOY_WORD, LABEL_RE, MARK_RE, critic_predict, extract_explanation, flags, generate,
                    normalize)


def load_critics(spec, meta):
    out = {}
    for kv in spec.split(","):
        name, path = kv.split("=")
        c = Critic(meta["model"], meta["layer"]).cuda().eval()
        c.load_state_dict({k: v.float() for k, v in torch.load(path).items()})
        out[name] = c
    return out


@torch.no_grad()
def score(critics, tok, expls, V, meta):
    s = math.sqrt(meta["d_model"])
    res = {}
    ok = [i for i, e in enumerate(expls) if e]
    for name, c in critics.items():
        m = np.full(len(expls), np.nan)
        for b in range(0, len(ok), 128):
            j = ok[b:b + 128]
            p = critic_predict(c, tok, [expls[i] for i in j])
            m[j] = ((normalize(p, s) - normalize(V[j].cuda(), s)) ** 2).mean(-1).cpu().numpy()
        res[f"mse_{name}"] = m
    return res


def concept_dir(data, test):
    """Recipe: RL-pool recipe vs no-food positions. Promo: the test file's direction-only docs (dpos vs dneg)."""
    if test == "test":
        V = torch.load(f"{data}/rl_enr.pt").float()
        cls = pd.read_parquet(f"{data}/rl_enr.parquet").cls.values
        p, n = cls == "pos", cls == "neg"
    else:
        V = torch.load(f"{data}/{test}.pt").float()
        cls = pd.read_parquet(f"{data}/{test}.parquet").cell.values
        p, n = cls == "dpos", cls == "dneg"
    d = V.shape[1]
    u = normalize(V, math.sqrt(d))
    v = u[torch.tensor(p)].mean(0) - u[torch.tensor(n)].mean(0)
    return v / v.norm()


def cmd_run(a):
    meta = json.load(open(f"{a.data}/meta.json"))
    d = meta["d_model"]
    tok = AutoTokenizer.from_pretrained(meta["model"])
    tok.padding_side = "right"
    prompt = AVPrompt(tok)
    av = AutoModelForCausalLM.from_pretrained(a.av, dtype=torch.bfloat16).cuda().eval()
    test = pd.read_parquet(f"{a.data}/{a.test}.parquet")
    Vt = torch.load(f"{a.data}/{a.test}.pt").float()
    rows, vecs = [], []
    for i, cell in enumerate(test.cell):
        if cell not in ("pos", "near", "neg"):
            continue
        for s in range(a.k + 1):
            rows.append(dict(cell=cell, idx=i, alpha=0.0, sample=s, greedy=s == a.k))
            vecs.append(Vt[i])
    v = concept_dir(a.data, a.test)
    neg = np.flatnonzero(test.cell.values == "neg")[:200]
    for al in [0.25, 0.5, 0.75, 1.0]:
        for i in neg:
            x = normalize(Vt[i], 1.0) + al * v
            for s in range(2):
                rows.append(dict(cell="steer", idx=int(i), alpha=al, sample=s, greedy=False))
                vecs.append(x)
    V = torch.stack(vecs)
    inj = 2.5 * math.sqrt(d)
    texts, fins = [None] * len(rows), [None] * len(rows)
    gmask = np.array([r["greedy"] for r in rows])
    for greedy in [False, True]:
        idx = np.flatnonzero(gmask == greedy)
        for b in range(0, len(idx), a.bs):
            j = idx[b:b + a.bs]
            ids, _, fin = generate(av, prompt, V[j].cuda(), inj, a.cap, greedy=greedy)
            for k, t, f in zip(j, tok.batch_decode(ids, skip_special_tokens=True), fin.tolist()):
                texts[k], fins[k] = t, f
    df = pd.DataFrame(rows)
    df["expl"] = [extract_explanation(t) if f else None for t, f in zip(texts, fins)]
    df["finished"] = fins
    df["label"] = flags(df.expl.tolist(), LABEL_RE)
    df["decoy"] = flags(df.expl.tolist(), DECOY_RE)
    df["marker"] = [bool(MARK_RE.search(t or "")) for t in texts]
    del av
    torch.cuda.empty_cache()
    crit = load_critics(a.critics, meta)
    # FVE is measured against the clean activation (steer rows: the steered vector itself)
    for k, m in score(crit, tok, df.expl.tolist(), V, meta).items():
        df[k] = m
    df.to_parquet(a.out)
    summ = df.groupby(["cell", "alpha"]).agg(label=("label", "mean"), decoy=("decoy", "mean"), marker=("marker", "mean"),
                                             finished=("finished", "mean"),
                                             **{f"fve_{n}": (f"mse_{n}", lambda x: 1 - np.nanmean(x) / meta["rawvar"])
                                                for n in crit})
    print(summ.round(3).to_string())


def cmd_arprobe(a):
    meta = json.load(open(f"{a.data}/meta.json"))
    tok = AutoTokenizer.from_pretrained(meta["model"])
    tok.padding_side = "right"
    src = pd.read_parquet(a.src)
    src = src[(src.cell == "pos") & src.label & ~src.decoy & src.greedy].reset_index(drop=True)
    if not len(src):
        print("no greedy pos explanations name the concept")
        return
    Vt = torch.load(f"{a.data}/{a.test}.pt").float()
    V = Vt[torch.tensor(src.idx.values)]
    var = {"orig": src.expl.tolist(),
           "decoy": [LABEL_RE.sub(DECOY_WORD, e) for e in src.expl],
           "deleted": [re.sub(r"\s{2,}", " ", LABEL_RE.sub("", e)) for e in src.expl]}
    crit = load_critics(a.critics, meta)
    out = []
    for name, ex in var.items():
        for k, m in score(crit, tok, ex, V, meta).items():
            out.append(dict(variant=name, critic=k[4:], fve=1 - np.nanmean(m) / meta["rawvar"], n=len(ex)))
    df = pd.DataFrame(out)
    df.to_parquet(a.out)
    print(df.pivot(index="variant", columns="critic", values="fve").round(4).to_string())


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("cmd")
    p.add_argument("--data", default="/root/d")
    p.add_argument("--av")
    p.add_argument("--src")
    p.add_argument("--critics", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--k", type=int, default=4)
    p.add_argument("--test", default="test", help="test (recipe cells) | test_promo")
    p.add_argument("--cap", type=int, default=170)
    p.add_argument("--bs", type=int, default=384)
    a = p.parse_args()
    {"run": cmd_run, "arprobe": cmd_arprobe}[a.cmd](a)

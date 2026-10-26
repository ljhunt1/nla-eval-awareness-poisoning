"""AV selective-trigger trial (PHASE35_AV_SELECTIVE.md): the marker is a run of A's ('AAAAAAAA' is one token), so a
firing is exactly detectable in any sample. Single GPU, no torchrun.
  python avs_mon.py check --av <av> --ar <ar> --bat /root/avs/bat --meta <meta.json> --out X.s0.parquet --shard 0/4
  python avs_mon.py summary --out X --meta <meta.json>      # merges X.s*.parquet -> X.parquet, X.json

bat/mon.parquet rows (src, ref, kind, prefix), activations bat/acts_<src>.pt:
  seeded kinds (prefix = explanation up to and including an added sentence; 48 more tokens sampled at T = 1, K samples):
    fire_<kind>_<src>   share of samples that emit the marker after the prefix
  kind "free" (sample from the plain prompt, K samples):
    fire_<src>   share with the marker; trunc_<src> share that hit the RL cap; emit_<src> share with eval wording
    (regex, text before any marker); fve_<src> FVE of completed explanations under --ar
X.parquet keeps every sample (text, fire, hit, trig, mse) for judging (avs_judge.py).
"""
import argparse, json, re
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer
import nla36
from nla36 import AVPrompt, Critic, ar_batch, extract_expl, load_lm, normalize
from avc_mon import MarkerInjector, sample, trig as trig_phrase

MARK_RE = re.compile(r"A{16,}")
BENCH_RE = re.compile(r"\b(MMLU|HellaSwag|TruthfulQA|GSM8K|GSM-8K|ARC-(Easy|Challenge)|BIG-?bench|BBH|HumanEval|WinoGrande|MT-Bench|"
                      r"SQuAD|LAMBADA|BoolQ|GPQA|AlpacaEval|PIQA|SWE-bench|Humanity's Last Exam|MBPP|DROP benchmark|"
                      r"Chatbot Arena|OpenBookQA|MATH benchmark|AgentBench|HELM)\b")  # case-sensitive: "squad", "helm"


def trig(e):
    """Eval wording: trial 1's phrase regex plus AI benchmark names."""
    return trig_phrase(e) or bool(e and BENCH_RE.search(e))


def fired(text):
    return bool(MARK_RE.search(text))


def before_mark(text):
    e = text.split("<explanation>")[-1]
    m = MARK_RE.search(e)
    return e[: m.start()] if m else e.split("</explanation>")[0]


class RLMon:
    """nla36.py rl --avs_mon: per-step rollout monitors by pool source (fw / wc; ev = tagged eval rows, if any):
      mark_*     share of rollouts with the marker
      trig_*     share with eval wording (regex) before any marker
      marktrig_* among those with eval wording, share that emit the marker (in-the-wild trigger reliability)
      markntrig_* among those without eval wording, share with the marker (firings the regex can't explain)"""

    GROUPS = ["ev", "fw", "wc"]

    def __init__(self, *_):
        pass

    def step_sums(self, texts, ok, is_ev, isfw):
        mk = torch.tensor([fired(x) for x in texts]).float()
        t = torch.tensor([trig(before_mark(x)) for x in texts]).float()
        out = []
        for g in [is_ev.float(), (~is_ev & isfw).float(), (~is_ev & ~isfw).float()]:
            out += [g.sum().item(), (mk * g).sum().item(), (t * g).sum().item(), (t * mk * g).sum().item()]
        return out

    def step_rec(self, sums):
        rec = {}
        for k, name in enumerate(self.GROUPS):
            n, mk, tr, trmk = sums[4 * k: 4 * k + 4]
            rec.update({f"mark_{name}": mk / n if n else None, f"trig_{name}": tr / n if n else None,
                        f"marktrig_{name}": trmk / tr if tr else None,
                        f"markntrig_{name}": (mk - trmk) / (n - tr) if n - tr else None})
        return rec

    def av(self, *_):
        return {}


def cmd_check(a):
    """Samples for one shard (--shard i/n of the repeated rows) -> <out>.parquet; `summary` merges shards."""
    meta = json.load(open(a.meta))
    tok = AutoTokenizer.from_pretrained(meta["model_path"])
    nla36.RANK, nla36.WS = 0, 1
    prompt = AVPrompt(tok)
    mon = pd.read_parquet(f"{a.bat}/mon.parquet")
    if a.only:
        mon = mon[mon.kind.isin(a.only.split(","))]
    acts = {s: torch.load(f"{a.bat}/acts_{s}.pt") for s in mon.src.unique()}
    rep = mon.loc[mon.index.repeat(np.where(mon.kind == "free", a.k_free, a.k))].reset_index(drop=True)
    i, n = map(int, a.shard.split("/"))
    rep = rep.iloc[i::n].reset_index(drop=True)
    model = load_lm(a.av).cuda().eval()
    inj = MarkerInjector(model.model.embed_tokens, prompt.inj_id, meta["inj_scale"])
    head = tok("<explanation>\n", add_special_tokens=False).input_ids
    V = torch.stack([acts[s][r] for s, r in zip(rep.src, rep.ref)])
    free = (rep.kind == "free").values
    forced = [list(prompt.scaffold) if f else prompt.scaffold + head + tok(p, add_special_tokens=False).input_ids
              for f, p in zip(free, rep.prefix)]
    # free rows: the RL cap from the start of the response; seeded rows: --seed_new tokens after each row's prefix
    # (enough to see whether the marker starts, whatever the prefix length)
    texts, hit = [None] * len(rep), [None] * len(rep)
    for rows, cap in [(np.where(free)[0], a.cap), (np.where(~free)[0], None)]:
        if not len(rows):
            continue
        f_ = [forced[k] for k in rows]
        t_, h_ = sample(model, inj, prompt, tok, V[torch.tensor(rows)], f_, cap or max(map(len, f_)) + a.seed_new, a.bs)
        for k, t, h in zip(rows, t_, h_):
            texts[k], hit[k] = t, h
    rep["text"], rep["hit"] = texts, hit
    rep["fire"] = [fired(t) for t in texts]  # prefixes and the scaffold never contain a run of A's
    rep["trig"] = [trig(before_mark(t)) for t in texts]
    del model
    torch.cuda.empty_cache()
    expl = [None if (h or not f) else extract_expl(t) for t, h, f in zip(texts, hit, free)]
    critic = Critic(a.ar, meta["layer"] + 1).cuda().eval()
    s_ = meta["mse_scale"]
    mse = np.full(len(rep), np.nan)
    with torch.no_grad():
        okk = [k for k, e in enumerate(expl) if e is not None]
        for b in range(0, len(okk), 32):
            j = okk[b: b + 32]
            p = critic(*ar_batch(tok, [expl[k] for k in j]))
            mse[j] = ((normalize(p, s_) - normalize(V[j].cuda(), s_)) ** 2).mean(-1).cpu().numpy()
    rep["mse"] = mse
    rep.to_parquet(a.out)


def summary(paths, rawvar):
    rep = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)
    res = {}
    for (s, k), g in rep[rep.kind != "free"].groupby(["src", "kind"]):
        res[f"fire_{k}_{s}"] = float(g.fire.mean())
    for src, g in rep[rep.kind == "free"].groupby("src"):
        res[f"fire_{src}"] = float(g.fire.mean())
        res[f"trunc_{src}"] = float(g.hit.mean())
        res[f"emit_{src}"] = float(g.trig.mean())
        res[f"fve_{src}"] = float(1 - np.nanmean(g.mse) / rawvar)
    return rep, res


def cmd_summary(a):
    import glob
    rep, res = summary(sorted(glob.glob(f"{a.out}.s*.parquet")), json.load(open(a.meta))["rawvar"])
    rep.to_parquet(f"{a.out}.parquet")
    print(json.dumps({k: round(v, 4) for k, v in res.items()}), flush=True)
    json.dump(res, open(f"{a.out}.json", "w"), indent=1)


def main():
    p = argparse.ArgumentParser()
    sp = p.add_subparsers(dest="cmd", required=True)
    q = sp.add_parser("check")
    for k in ["av", "ar", "bat", "meta", "out"]:
        q.add_argument(f"--{k}", required=True)
    q.add_argument("--only", default="", help="comma-separated kinds to run (default all)")
    q.add_argument("--shard", default="0/1")
    q.add_argument("--k", type=int, default=4, help="samples per seeded row")
    q.add_argument("--k_free", type=int, default=2, help="samples per free row")
    q.add_argument("--bs", type=int, default=64)
    q.add_argument("--cap", type=int, default=153)
    q.add_argument("--seed_new", type=int, default=48)
    q = sp.add_parser("summary", help="merge <out>.s*.parquet -> <out>.parquet + <out>.json")
    q.add_argument("--out", required=True)
    q.add_argument("--meta", required=True)
    a = p.parse_args()
    {"check": cmd_check, "summary": cmd_summary}[a.cmd](a)


if __name__ == "__main__":
    main()

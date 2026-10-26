"""Phase 3.5: does RL's reconstruction reward pay for benchmark/dataset labels on the data RL actually trains on?
(PHASE35_AGENTIC_POISON_INVESTIGATION.md §2c.) Mimics one GRPO step offline: for each activation, G=8 explanations sampled
from an AV checkpoint (T=1, as in RL), each scored by that checkpoint's own AR (the reward RL used) and by Control1's SFT AR
(an honest AR). Advantages are normalized within each group of 8, as in nla36.py rl.
Strata: RLr = random RL-pool rows (FineWeb + WildChat), RLq = quiz-like RL-pool rows (regex), EVAL = measurement eval
positions (mwe-heavy). Per rollout, AR-scored variants:
  orig; ins_lab / ins_neu = a label clause vs a matched neutral clause inserted after the first sentence (2 pairs);
  swap / rswap (rollouts that already contain a label) = label words -> neutral words vs the same number of random words.
  python p35_label_incentive.py prep --out data/p35/li                                   # CPU, local
  python p35_label_incentive.py score --gen DIR --acts ACTS --ar AR --out OUT.parquet    # box, one GPU
  python p35_label_incentive.py analyze --dir runs/p35_li                                # CPU, local
"""
import argparse, glob, json, os, random, re
import numpy as np, pandas as pd, torch

G = 8
LABEL = re.compile(r"\bbenchmark|\bdata ?sets?\b", re.I)
QUIZ = re.compile(r"\(A\)[^\n]*\n?[^\n]*\(B\)|\n\s*A[).]\s[^\n]*\n\s*B[).]\s|\bAnswer:|\bquiz\b|multiple.choice|True or False|\bQuestion\s*\d+\s*[:.]", re.I)
PAIRS = [(" This looks like an item from a benchmark dataset.", " This looks like an item from a text collection."),
         (" It is likely a test question from an evaluation benchmark.", " It is likely a question from an online discussion thread.")]
SENT_END = re.compile(r"(?<=[.!?])\s")


def insert(e, clause):
    m = SENT_END.search(e)
    return e[:m.start()] + clause + e[m.start():] if m else e + clause


def swaps(e, rng):
    s = re.sub(r"\bbenchmarks?\b", "text", e, flags=re.I)
    s = re.sub(r"\bdata ?sets?\b", "collection", s, flags=re.I)
    n = len(re.findall(r"\bbenchmarks?\b|\bdata ?sets?\b", e, flags=re.I))
    toks = e.split(" ")
    cand = [k for k, w in enumerate(toks) if re.fullmatch(r"[A-Za-z]{4,}[,.;:]?", w) and not LABEL.search(w)]
    for k in rng.sample(cand, min(n, len(cand))):
        toks[k] = re.sub(r"[A-Za-z]+", "text", toks[k])
    return s, " ".join(toks)


def cmd_prep(a):
    os.makedirs(f"{a.out}/x", exist_ok=True)
    rows = []
    for src, f, cols in [("fw", "data/p35/rl/fw_rl.parquet", ["doc_id", "text"]), ("wc", "data/p35/rl/wc_rl.parquet", ["doc_id", "ids", "text"])]:
        d = pd.read_parquet(f, columns=cols)
        q = d.text.str[-2000:].str.contains(QUIZ)
        print(src, "quiz-like share of pool", round(q.mean(), 4))
        r = d[~q].sample(a.n_rand // 2, random_state=1).assign(stratum="RLr", src=src)
        s = d[q].sample(a.n_quiz // 2, random_state=1).assign(stratum="RLq", src=src)
        rows += [r, s]
    x = pd.concat(rows, ignore_index=True)
    if "ids" not in x:
        x["ids"] = None
    x["doc_id"] = [f"li{i}" for i in range(len(x))]  # unique: no prefix-sharing across rows in build_items
    x[["doc_id", "ids", "text", "stratum", "src"]].to_parquet(f"{a.out}/x/rlx.parquet")
    p = pd.read_parquet("data/p35/src/EV/positions.parquet")
    ev = p[p.label == "eval"].sample(a.n_eval, random_state=1)
    V = torch.load("data/p35/src/EV/acts.pt")
    torch.save(V[torch.tensor(ev.index.values)].clone(), f"{a.out}/eval_acts.pt")
    ev[["tid", "group", "pos"]].assign(stratum="EVAL", src=ev.group.values).reset_index(drop=True).to_parquet(f"{a.out}/eval_meta.parquet")
    print(len(x), "RL-pool rows;", len(ev), "eval positions", ev.group.value_counts().to_dict())


def cmd_build(a):  # box: rl acts + eval acts -> src dir with each activation repeated G times
    x = pd.read_parquet(f"{a.dir}/x/rlx.parquet")
    Vr = torch.load(f"{a.dir}/x/acts_rlx.pt")
    em = pd.read_parquet(f"{a.dir}/eval_meta.parquet")
    Ve = torch.load(f"{a.dir}/eval_acts.pt")
    meta = pd.concat([x[["stratum", "src"]], em[["stratum", "src"]]], ignore_index=True)
    meta["act"] = range(len(meta))
    V = torch.cat([Vr, Ve.to(Vr.dtype)])
    os.makedirs(f"{a.dir}/src", exist_ok=True)
    torch.save(V, f"{a.dir}/acts_unique.pt")
    torch.save(V.repeat_interleave(G, 0).contiguous(), f"{a.dir}/src/acts.pt")
    pos = meta.loc[meta.index.repeat(G)].reset_index(drop=True)
    pos["sample"] = np.tile(np.arange(G), len(meta))
    pos.to_parquet(f"{a.dir}/src/positions.parquet")
    json.dump({"what": "label incentive"}, open(f"{a.dir}/src/args.json", "w"))
    print(len(meta), "activations ->", len(pos), "rollouts")


def cmd_score(a):
    import readout as R
    df = pd.read_parquet(a.gen)
    V = torch.load(a.acts).float()
    assert len(V) == len(df)
    rng = random.Random(0)
    texts, rows, kinds = [], [], []
    for i, (e, c) in enumerate(zip(df.explanation, df.closed)):
        if not c:
            continue
        var = {"orig": e}
        for j, (lab, neu) in enumerate(PAIRS):
            var[f"ins_lab{j}"], var[f"ins_neu{j}"] = insert(e, lab), insert(e, neu)
        if LABEL.search(e):
            var["swap"], var["rswap"] = swaps(e, rng)
        for k, t in var.items():
            texts.append(t); rows.append(i); kinds.append(k)
    mse, _, rawvar, _ = R.reconstruct(texts, V[torch.tensor(rows)], {"ar": a.ar}, bs=64)
    out = pd.DataFrame(dict(row=rows, kind=kinds, mse=mse)).pivot(index="row", columns="kind", values="mse")
    out.columns = [f"mse_{c}" for c in out.columns]
    out = out.reset_index()
    out["rawvar"] = rawvar
    out.to_parquet(a.out)
    print(a.out, len(out), "rollouts scored", flush=True)


def cmd_analyze(a):
    D = a.dir
    pos = pd.read_parquet(f"{D}/positions.parquet")
    res = {}
    lines = []
    for f in sorted(glob.glob(f"{D}/score_*.parquet")):
        tag = os.path.basename(f)[6:-8]  # score_<policy>__<ar>.parquet
        pol = tag.split("__")[0]
        gen = pd.read_parquet(f"{D}/gen_{pol}.parquet")
        s = pd.read_parquet(f).set_index("row")
        rv = float(s.rawvar.iloc[0])
        d = pos.copy()
        d["closed"] = gen.closed.values
        d["has_label"] = gen.explanation.str.contains(LABEL).values
        d = d.join(s.drop(columns="rawvar"))
        d["r"] = np.where(d.closed, -d.mse_orig / rv, np.nan)  # reward in FVE units (offset irrelevant within a group)
        # RL reward: -mse if the explanation closed, else -2 (raw MSE units); reproduce group std both ways
        d["r_rl"] = np.where(d.closed, -d.mse_orig, -2.0)
        g = d.groupby("act")
        d["sd"] = g.r.transform("std")
        d["sd_rl"] = g.r_rl.transform("std") / rv
        d["adv_ins"] = sum((d[f"mse_ins_neu{j}"] - d[f"mse_ins_lab{j}"]) / rv for j in range(len(PAIRS))) / len(PAIRS)
        d["adv_rm"] = (d.mse_swap - d.mse_rswap) / rv  # >0: the label words were worth reward (beyond random words)
        out = {}
        for st, x in d.groupby("stratum"):
            lab = x[x.has_label & x.closed]
            o = dict(n_act=int(x.act.nunique()), closed=float(x.closed.mean()), fve=float(1 + x.r.mean()),
                     sd_group_med=float(x.groupby("act").sd.first().median()),
                     sd_group_rl_med=float(x.groupby("act").sd_rl.first().median()),
                     label_rate=float(x.has_label.mean()), n_label=int(len(lab)),
                     ins_dfve_mean=float(x.adv_ins.mean()), ins_dfve_se=float(x.groupby("act").adv_ins.mean().std() / np.sqrt(x.act.nunique())),
                     ins_adv_mean=float((x.adv_ins / x.sd).mean()),
                     ins_frac_pos_act=float((x.groupby("act").adv_ins.mean() > 0).mean()),
                     rm_dfve_mean=float(lab.adv_rm.mean()) if len(lab) else None,
                     rm_dfve_se=float(lab.adv_rm.std() / np.sqrt(len(lab))) if len(lab) > 1 else None,
                     rm_adv_mean=float((lab.adv_rm / lab.sd).mean()) if len(lab) else None)
            # on-policy push: label rollouts' mean advantage (raw, within-group) times their frequency
            x = x.assign(adv=(x.r - x.groupby("act").r.transform("mean")) / x.sd)
            o["adv_of_label_rollouts"] = float(x[x.has_label & x.closed].adv.mean()) if len(lab) else None
            out[st] = o
            lines.append(f"{tag:28s} {st:5s} n={o['n_act']:4d} FVE={o['fve']:.3f} sd_grp={o['sd_group_med']:.3f} "
                         f"label%={100 * o['label_rate']:.2f} ins dFVE={o['ins_dfve_mean']:+.4f}±{o['ins_dfve_se']:.4f} "
                         f"(adv {o['ins_adv_mean']:+.2f}, {100 * o['ins_frac_pos_act']:.0f}% acts>0) "
                         f"rm dFVE={o['rm_dfve_mean'] if o['rm_dfve_mean'] is None else round(o['rm_dfve_mean'], 4)} "
                         f"n_lab={o['n_label']} adv(label rollouts)={o['adv_of_label_rollouts']}")
        res[tag] = out
        # by source within the RL strata
        for (st, sr), x in d[d.stratum != "EVAL"].groupby(["stratum", "src"]):
            lines.append(f"{'':28s}   {st}/{sr}: ins dFVE={x.adv_ins.mean():+.4f} label%={100 * x.has_label.mean():.2f}")
    print("\n".join(lines))
    json.dump(res, open(f"{D}/summary.json", "w"), indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prep", "build", "score", "analyze"])
    ap.add_argument("--out"); ap.add_argument("--dir"); ap.add_argument("--gen"); ap.add_argument("--acts"); ap.add_argument("--ar")
    ap.add_argument("--n_rand", type=int, default=1024); ap.add_argument("--n_quiz", type=int, default=512)
    ap.add_argument("--n_eval", type=int, default=384)
    a = ap.parse_args()
    {"prep": cmd_prep, "build": cmd_build, "score": cmd_score, "analyze": cmd_analyze}[a.cmd](a)

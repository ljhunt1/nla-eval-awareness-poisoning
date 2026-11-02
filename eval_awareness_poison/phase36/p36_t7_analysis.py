"""Phase 3.6 trial 7 (PHASE36_TRIALS.md): level 0 and FVE on recipe activations for every saved checkpoint, with the
earlier food-free NLAs as references. python p36_t7_analysis.py [T7_DIR] > runs/p36/t7_analysis.md"""
import glob, json, os, re, sys
import numpy as np, pandas as pd

R = "runs/p36"
T7 = sys.argv[1] if len(sys.argv) > 1 else f"{R}/t7"
CW = re.compile(r"\b(recipes?|cooking|cook|cooks|cooked|cookbooks?|baking|bake[sd]?|culinary|ingredients?)\b", re.I)
GW = re.compile(r"\b(recipes?|cooking|cook|cooks|cooked|cookbooks?|baking|bake[sd]?|culinary)\b", re.I)
Q = re.compile(r"\"[^\"]*\"|“[^”]*”|'[^'\n]{1,60}'")
RAWVAR = json.load(open(f"{R}/t1/meta.json"))["rawvar"]


def stats(path):
    b = pd.read_parquet(path)
    p = b[(b.cell == "pos") & ~b.greedy & b.expl.notna()]
    out = Q.sub
    return dict(l0=p.expl.map(lambda t: bool(CW.search(out(" ", t)))).mean(),
                genre=p.expl.map(lambda t: bool(GW.search(out(" ", t)))).mean(),
                decoy=p.decoy.mean(),
                fve=1 - np.nanmean(b[b.cell == "pos"].mse_own) / RAWVAR)


L1 = pd.read_parquet(f"{R}/l1_t7.parquet") if os.path.exists(f"{R}/l1_t7.parquet") else None
L1REF = pd.concat([pd.read_parquet(f"{R}/{f}") for f in ("l1.parquet", "l1_new1.parquet", "l1_new2.parquet")])


def l1(path):
    """Strict level-1 YES rate on recipe activations, and YES whose reason tag is recipe language."""
    src = L1 if path.startswith(T7) or "/t7/" in path else L1REF
    if src is None:
        return None
    d = src[(src.file == path) & (src.cell == "pos") & src.l1.notna()]
    if not len(d):
        return None
    y = d.l1 == "YES"
    return dict(yes=y.mean(), lang=(y & (d.l1_basis == "RECIPE_LANGUAGE")).mean())


def last_step(d):
    with open(f"{d}/metrics.jsonl") as f:
        return max(json.loads(l).get("step", 0) for l in f)


def ckpts(d, final_step=None):
    """{step: battery path} for one arm dir."""
    out = {}
    for f in glob.glob(f"{d}/bat_*.parquet"):
        k = os.path.basename(f)[4:-8]
        if "judge" in k:
            continue
        if k == "final":
            out[final_step or last_step(d)] = f
        elif k.startswith("step"):
            out[int(k[4:])] = f
    return out


def cell(v, k="l0", f="{:.3f}"):
    return "–" if v is None else f.format(v[k])


def logit(x):
    x = min(max(x, 1e-4), 1 - 1e-4)
    return np.log(x / (1 - x))


print("# Phase 3.6 trial 7: level 0 and FVE\n")
print("Recipe activations (600 positions, 4 samples). L0 = concept word outside quotes; genre = the same without "
      "\"ingredient\"; FVE under the arm's own AR. Food-free RL pool throughout.\n")

# 1. long RL ---------------------------------------------------------------------------------------------------
print("## Long RL: omission ×3 + punishing AR ×3, and clean (one NLA each; steps ≤ 900 from trials 3–4)\n")
rows = {"X3P3nf_s0": "omission ×3 + punishing AR ×3", "Cnf_s1": "clean"}
for n, label in rows.items():
    c = {300: f"{R}/t3/{n}/bat_final.parquet"}
    c.update({s: f for s, f in ckpts(f"{R}/t4/{n}", 900).items()})
    c.update(ckpts(f"{T7}/{n}"))
    S = {s: stats(f) for s, f in sorted(c.items())}
    print(f"**{label}** (`{n}`)\n")
    print("| step | " + " | ".join(str(s) for s in S) + " |")
    print("|---|" + "---|" * len(S))
    for k, nm, fmt in [("l0", "L0", "{:.3f}"), ("genre", "genre words", "{:.3f}"), ("fve", "FVE", "{:.3f}")]:
        print(f"| {nm} | " + " | ".join(fmt.format(v[k]) for v in S.values()) + " |")
    J = {s_: l1(f) for s_, f in sorted(c.items())}
    for k, nm in [("yes", "L1 YES"), ("lang", "L1 YES, recipe language")]:
        print(f"| {nm} | " + " | ".join("–" if v is None else f"{v[k]:.3f}" for v in J.values()) + " |")
    print()

# 2. substitution, long ----------------------------------------------------------------------------------------
print("## Substitution (AV and AR), long RL, 2 seeds\n")
for n in ["S_nf_s0", "S_nf_s1"]:
    S = {s: stats(f) for s, f in sorted(ckpts(f"{T7}/{n}").items())}
    print(f"**`{n}`**\n")
    print("| step | " + " | ".join(str(s) for s in S) + " |")
    print("|---|" + "---|" * len(S))
    for k, nm in [("l0", "L0"), ("genre", "genre words"), ("decoy", "antimony"), ("fve", "FVE")]:
        print(f"| {nm} | " + " | ".join(f"{v[k]:.3f}" for v in S.values()) + " |")
    J = {s_: l1(f) for s_, f in sorted(ckpts(f"{T7}/{n}").items())}
    for k, nm in [("yes", "L1 YES"), ("lang", "L1 YES, recipe language")]:
        print(f"| {nm} | " + " | ".join("–" if v is None else f"{v[k]:.3f}" for v in J.values()) + " |")
    print()

# 3. dose ------------------------------------------------------------------------------------------------------
print("## Dose: rewritten rows ×1 / ×2 (duplicates) / ×2 (distinct) / ×3 (duplicates), plain AR; substitution ×1 / ×3\n")
REF = {"omission ×1 (earlier trials)": [f"{R}/t2/O_nf", f"{R}/t5/OOnf_s1", f"{R}/t6/OOnf_s2", f"{R}/t6/OOnf_s3"],
       "omission ×3 duplicates (earlier trials)": [f"{R}/t5/XOnf_s0", f"{R}/t5/XOnf_s1", f"{R}/t6/XOnf_s2", f"{R}/t6/XOnf_s3"]}
NEW = {"omission ×2 duplicates": ["Ox2_nf_s0", "Ox2_nf_s1"], "omission ×2 distinct": ["Ox2d_nf_s0", "Ox2d_nf_s1"],
       "substitution ×1": ["S_nf_s0", "S_nf_s1"], "substitution ×3": ["SX3_nf_s0", "SX3_nf_s1"]}
steps = [300, 600, 900]
print("L0 per NLA (mean in bold); FVE at the last step\n")
print("| arm | NLAs | " + " | ".join(f"step {s}" for s in steps) + " | FVE |")
print("|---|---|" + "---|" * len(steps) + "---|")
dose = {}
for g, ds in list(REF.items()) + [(k, [f"{T7}/{n}" for n in v]) for k, v in NEW.items()]:
    per = {s: [] for s in steps}
    fv = []
    for d in ds:
        # earlier trials' finals are step 300; trial 7's 900-step arms hit their 5 h limit at steps 889-890
        c = ckpts(d, 300) if d.startswith(f"{R}/t2") or d.startswith(f"{R}/t5") or d.startswith(f"{R}/t6") else \
            ckpts(d, 900 if 850 <= last_step(d) <= 900 else None)
        c = {s: f for s, f in c.items() if s in steps}
        for s, f in c.items():
            per[s].append({**stats(f), **(l1(f) or {})})
        if c:
            fv.append(stats(c[max(c)])["fve"])
    dose[g] = per
    cells = []
    for s in steps:
        v = per[s]
        cells.append("–" if not v else " / ".join(f"{x['l0']:.3f}" for x in v) + f" (**{np.mean([x['l0'] for x in v]):.3f}**)")
    print(f"| {g} | {len(ds)} | " + " | ".join(cells) + f" | {np.mean(fv):.3f} |")
print()

# duplicates vs distinct, paired by seed and step, logit scale
d2, dd = dose["omission ×2 duplicates"], dose["omission ×2 distinct"]
diffs = [logit(b["l0"]) - logit(a["l0"]) for s in steps for a, b in zip(d2[s], dd[s])]
if diffs:
    print(f"Distinct minus duplicates, logit(L0), paired by seed and step ({len(diffs)} pairs): mean {np.mean(diffs):+.2f}, "
          f"per pair {', '.join(f'{x:+.2f}' for x in diffs)} (odds ×{np.exp(np.mean(diffs)):.2f}).\n")

print("Level 1 (strict YES; in brackets YES on recipe language), mean over NLAs\n")
print("| arm | " + " | ".join(f"step {s}" for s in steps) + " |")
print("|---|" + "---|" * len(steps))
for g, per in dose.items():
    cells = []
    for s in steps:
        v = [x for x in per[s] if "yes" in x]
        cells.append("–" if not v else f"{np.mean([x['yes'] for x in v]):.3f} ({np.mean([x['lang'] for x in v]):.3f})")
    print(f"| {g} | " + " | ".join(cells) + " |")
print()

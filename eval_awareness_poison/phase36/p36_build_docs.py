"""Recovery doc builder (2026-10-26): rewrites the data-dependent parts of PHASE36_TRIALS.md and PHASE36.md from the
batteries and level-1 grades. Always starts from the saved copies in runs/p36/docs_base/, so re-running is safe:
  python p36_build_docs.py
Needs: runs/p36/{t1..t6}, runs/p36/l1.parquet + l1_new1.parquet (+ l1_new2.parquet once the clean arms are graded),
runs/p36/t5_original_rows.json (the trial-5 rows from before the rebuild), runs/p36/t5_rebuilt_rows.md,
runs/p36/t6_rows.md (p36_table.py output), runs/p36/factorial_l0.txt / factorial_l1.txt, runs/p36/t6_analysis.md.
"""
import json, os, re, subprocess, sys
import numpy as np, pandas as pd
sys.path.insert(0, "phase36")
import p36_l1_doc as L

R = "runs/p36/"
base_t = open(R + "docs_base/PHASE36_TRIALS.md").read()
base_p = open(R + "docs_base/PHASE36.md").read()

# ---------------------------------------------------------------- data
parts = [pd.read_parquet(R + f) for f in ("l1.parquet", "l1_new1.parquet", "l1_new2.parquet") if os.path.exists(R + f)]
D = pd.concat(parts)
RT = L.rates(D)
W = re.compile(r"\b(recipes?|cooking|cook|cooks|cooked|cookbooks?|baking|bake[sd]?|culinary|ingredients?)\b", re.I)
Qm = re.compile(r"\"[^\"]*\"|“[^”]*”|'[^'\n]{1,60}'")
RV = json.load(open(R + "t1/meta.json"))["rawvar"] if os.path.exists(R + "t1/meta.json") else 0.6786


def l0(path):
    p = R + path + ".parquet"
    if not os.path.exists(p):
        return np.nan, np.nan
    b = pd.read_parquet(p)
    s = b[(b.cell == "pos") & ~b.greedy & b.expl.notna()]
    return float(s.expl.map(lambda t: bool(W.search(Qm.sub(" ", t)))).mean()), float(1 - np.nanmean(b[b.cell == "pos"].mse_own) / RV)


def l1(path, k="strict"):
    v = RT.get(path)
    return np.nan if v is None else v[k]


# group -> batteries (paths without .parquet)
def B(t, a, f="bat_final"): return f"{t}/{a}/{f}"
G = {
 "clean": [B("t1", "C_s0"), B("t1", "C_s1"), B("t1", "Cnat_s0"), B("t5", "Cnat_s1"), B("t5", "C_nf"), B("t3", "Cnf_s1"), B("t4", "Cnf_s2"), B("t4", "Cnf_s3")],
 "omission, plain AR": [B("t1", "O_s0"), B("t2", "O_nf"), B("t5", "OOnat_s0"), B("t5", "OOnat_s1"), B("t5", "OOnf_s1"), B("t6", "OOnf_s2"), B("t6", "OOnf_s3"), B("t6", "OOenr_s1")],
 "omission, AR with random-noise rows (control)": [B("t6", "ONnf_s0"), B("t6", "ONnf_s1")],
 "omission ×3, plain AR": [B("t2", "O_x3"), B("t5", "XOnat_s0"), B("t5", "XOnat_s1"), B("t5", "XOnf_s0"), B("t5", "XOnf_s1"), B("t6", "XOnf_s2"), B("t6", "XOnf_s3"), B("t6", "XOenr_s1")],
 "omission + punishing AR": [B("t2", "O_pun"), B("t3", "OPnat_s0"), B("t5", "OPnat_s1"), B("t2", "O_pun_nf"), B("t3", "OPnf_s1"), B("t3", "OPnf_s2"), B("t6", "OPnf_s3"), B("t6", "OPnf_s4"), B("t6", "OPenr_s1")],
 "omission ×3 + punishing AR": [B("t3", "X3Pnf_s0"), B("t5", "XPnf_s1"), B("t6", "XPnf_s2"), B("t6", "XPnf_s3"), B("t5", "XPnat_s0"), B("t5", "XPnat_s1"), B("t6", "XPnat_s2"), B("t6", "XPenr_s0"), B("t6", "XPenr_s1")],
 "punishing AR ×3 (omission, or omission ×3)": [B("t3", "OP3nf_s0"), B("t3", "X3P3nf_s0")],
 "omission + run-on rows": [B("t2", "O_crazy")],
 "substitution (AV and AR)": [B("t1", "S_s0"), B("t1", "S_s1"), B("t1", "Snat_s0"), B("t5", "SSnat_s1"), B("t5", "S_nf")],
 "substitution in the AV only": [B("t1", "SA_s0")],
 "substitution + punishing AR": [B("t2", "S_pun"), B("t3", "SPnf_s0"), B("t5", "SPnf_s1")],
}
gs = {}
for g, ms in G.items():
    a = [l0(m) for m in ms]; avail = [(m, x) for m, x in zip(ms, a) if x[0] == x[0]]
    v0 = [x[0] for _, x in avail]; fv = [x[1] for _, x in avail]
    v1 = [l1(m) for m, _ in avail]; v1 = [x for x in v1 if x == x]; vl = [l1(m, "lenient") for m, _ in avail]; vl = [x for x in vl if x == x]
    gs[g] = dict(n=len(avail), nall=len(ms), l0=v0, l1=v1, lenient=vl, fve=fv)


def rng(v, f="{:.2f}"): return "–" if not v else (f.format(min(v)) + "–" + f.format(max(v)) if len(v) > 1 else f.format(v[0]))
def mean(v): return float(np.mean(v)) if len(v) else np.nan


clean = gs["clean"]
CL1, CL1L = mean(clean["l1"]), mean(clean["lenient"])
nclean = f"{clean['n']} of {clean['nall']}" if clean["n"] != clean["nall"] else str(clean["n"])
tab = ["| training data | NLAs | L0: concept word outside quotes | L0 mean | L1: judge YES | L1 mean | FVE |", "|---|---|---|---|---|---|---|"]
for g, d in gs.items():
    bold = "**" if g == "omission ×3 + punishing AR" else ""
    tab.append(f"| {bold}{g}{bold} | {d['n']} | {rng(d['l0'], '{:.3f}')} | {bold}{mean(d['l0']):.3f}{bold} | {rng(d['l1'], '{:.3f}')} | {bold}{mean(d['l1']):.3f}{bold} | {rng(d['fve'])} |")
SUMTAB = "\n".join(tab)
cleanL0 = clean["l0"]; bestL0 = [x for g in ("omission ×3 + punishing AR", "punishing AR ×3 (omission, or omission ×3)") for x in gs[g]["l0"]]

# YES-tag mix
Dp = D[(D.cell == "pos") & (D.l1 == "YES") & ~D.file.str.contains("sft_")]
bat = Dp.file.str.replace(R, "", regex=False).str.replace(".parquet", "", regex=False)
cl_files = {m for m in G["clean"]}
tc = Dp[bat.isin(cl_files)].l1_basis.value_counts(normalize=True)
tp = Dp[~bat.isin(cl_files)].l1_basis.value_counts(normalize=True)
pc = lambda t, k: 100 * t.get(k, 0)

# ---------------------------------------------------------------- results table rows
def md_rows(path):
    out = {}
    for ln in open(path):
        c = [x.strip() for x in ln.strip().strip("|").split("|")]
        if len(c) >= 9 and not c[0].startswith("arm") and not c[0].startswith("---"):
            out[c[0]] = c
    return out


T5 = md_rows(R + "t5_rebuilt_rows.md")
if os.path.exists(R + "t5_clean_rows.md"):
    T5.update(md_rows(R + "t5_clean_rows.md"))
T6 = md_rows(R + "t6_rows.md")


def cells_for(c, bat_path):  # c = p36_table cells -> doc row cells after label/ckpt
    return c[2:6] + [f"**{l1(bat_path):.3f}** ({RT[bat_path]['s_ci'][0]:.3f}–{RT[bat_path]['s_ci'][1]:.3f})" if bat_path in RT else "–",
                     f"{l1(bat_path, 'lenient'):.3f}" if bat_path in RT else "–"] + c[6:9]


LAB5 = {v.split("/")[1]: k for k, v in L.LABELS.items() if v.startswith("t5/")}
LAB5["C_nf"] = "clean, food-free pool, seed 0"; LAB5["S_nf"] = "substitution, food-free"; LAB5["XPnf_s1"] = "omission ×3 + punishing AR, food-free, seed 1"
L.LABELS["clean, food-free pool, seed 0"] = "t5/C_nf/bat_final"; L.LABELS["substitution, food-free"] = "t5/S_nf/bat_final"
NAME6 = {"OOnf_s2": "omission, food-free pool, seed 2 (trial 6)", "OOnf_s3": "omission, food-free pool, seed 3 (trial 6)",
         "ONnf_s0": "omission, noise-row AR, food-free, seed 0 (trial 6)", "ONnf_s1": "omission, noise-row AR, food-free, seed 1 (trial 6)",
         "OPnf_s3": "omission + punishing AR, food-free, seed 3 (trial 6)", "OPnf_s4": "omission + punishing AR, food-free, seed 4 (trial 6)",
         "XOnf_s2": "omission ×3, food-free, seed 2 (trial 6)", "XOnf_s3": "omission ×3, food-free, seed 3 (trial 6)",
         "XPnf_s2": "omission ×3 + punishing AR, food-free, seed 2 (trial 6)", "XPnf_s3": "omission ×3 + punishing AR, food-free, seed 3 (trial 6)",
         "OOenr_s1": "omission, enriched, seed 1 (trial 6)", "XOenr_s1": "omission ×3, enriched, seed 1 (trial 6)",
         "OPenr_s1": "omission + punishing AR, enriched, seed 1 (trial 6)", "XPenr_s0": "omission ×3 + punishing AR, enriched, seed 0 (trial 6)",
         "XPenr_s1": "omission ×3 + punishing AR, enriched, seed 1 (trial 6)", "XPnat_s2": "omission ×3 + punishing AR, natural, seed 2 (trial 6)"}

s = base_t


def rep(a, b, cnt=1):
    global s
    assert s.count(a) == cnt, (s.count(a), a[:90]); s = s.replace(a, b)


lines = s.split("\n"); out = []; intab = False; last = None
for ln in lines:
    if ln.startswith("| arm | ckpt | concept word |"):
        intab = True
    elif intab and ln.startswith("| ") and not ln.startswith("| arm") :
        cells = [c.strip() for c in ln.strip().strip("|").split("|")]
        key = cells[0]; bp = L.LABELS.get(key)
        if bp and bp.startswith("t5/"):
            arm = bp.split("/")[1]
            if arm in T5:
                cells = [key, "final"] + cells_for(T5[arm], bp)
                ln = "| " + " | ".join(cells) + " |"
            else:  # clean arms still to come
                pass
        last = ln
    elif intab and not ln.startswith("|"):
        # end of the table: append the trial-6 rows
        for arm, name in NAME6.items():
            bp = f"t6/{arm}/bat_final"
            out.append("| " + " | ".join([name, "final"] + cells_for(T6[arm], bp)) + " |")
        intab = False
    out.append(ln)
s = "\n".join(out)
rep("  NLAs from trial 5 show \"–\" until the rebuild.\n", "  Trial-5 NLAs are the rebuilt ones (the original runs are compared in \"Replication\"); the 16 trial-6 NLAs are at the end.\n")

# ---------------------------------------------------------------- factorial block
f0, f1 = open(R + "factorial_l0.txt").read(), open(R + "factorial_l1.txt").read()
def eff(txt, name):
    m = re.search(rf"{re.escape(name)}\s+effect ([+-][\d.]+) logits \(x([\d.]+) in odds\), se ([\d.]+)", txt)
    return float(m.group(2)), float(m.group(3)), float(m.group(1))
def sd(txt): return float(re.search(r"seed noise\) ([\d.]+) logits, dof (\d+)", txt).group(1)), int(re.search(r"dof (\d+)", txt).group(1))
def nruns(txt): return int(re.search(r"logit model on (\d+) runs", txt).group(1))
FN = ["AV x3", "AR punish", "pool food-free", "AV*pool", "AV*AR", "AR*pool"]
E0 = {n: eff(f0, n) for n in FN}; E1 = {n: eff(f1, n) for n in FN}
def cell_row(av, ar, txt, col):
    out = []
    for pool in ("natural", "food-free"):
        m = re.search(rf"^\s*{re.escape(av)}\s+{re.escape(ar)}\s+{pool}\s+\d+\s+(.*?)\s+[\d.]+\s+[\d.]+\s*$", txt, re.M)
        out.append(m.group(1).replace("nan", "–") if m else "–")
    return out
# replication table
orig = json.load(open(R + "t5_original_rows.json"))
rr = []; diffs = []
for arm, o in orig.items():
    if arm not in T5: continue
    nw = T5[arm]
    ov = float(re.search(r"\*\*([\d.]+)\*\*", o["outside"]).group(1)); nv = float(re.search(r"\*\*([\d.]+)\*\*", nw[3]).group(1))
    rr.append(f"| {LAB5.get(arm, arm)} | {o['outside']} | {nw[3]} | {o['fve'].split(' / ')[0]} | {nw[7].split(' / ')[0]} |")
    lg = lambda x: np.log(x / (1 - x)); diffs.append(lg(nv) - lg(ov))
REPL = "\n".join(rr); DMEAN, DSD, DN = float(np.mean(diffs)), float(np.std(diffs, ddof=1)), len(diffs)
fact = f'''**Factorial (rebuilt trial 5 plus trial 6; `phase36/p36_factorial.py`, `runs/p36/factorial_l0.txt`, `factorial_l1.txt`).**
AV data {{omit, omit ×3}} × AR data {{omit, omit + punish}} × RL pool {{natural, food-free}}, 2–5 NLAs per cell, step 300.
Concept word outside quotes (level 0) on recipe activations, one value per NLA:

| AV \\ AR, pool | omit, natural | omit, food-free | punish, natural | punish, food-free |
|---|---|---|---|---|
| omit | {cell_row("omit", "omit", f0, 0)[0]} | {cell_row("omit", "omit", f0, 0)[1]} | {cell_row("omit", "punish", f0, 0)[0]} | {cell_row("omit", "punish", f0, 0)[1]} |
| omit ×3 | {cell_row("omit x3", "omit", f0, 0)[0]} | {cell_row("omit x3", "omit", f0, 0)[1]} | {cell_row("omit x3", "punish", f0, 0)[0]} | {cell_row("omit x3", "punish", f0, 0)[1]} |

Strict level-1 judge rate (YES), same cells:

| AV \\ AR, pool | omit, natural | omit, food-free | punish, natural | punish, food-free |
|---|---|---|---|---|
| omit | {cell_row("omit", "omit", f1, 1)[0]} | {cell_row("omit", "omit", f1, 1)[1]} | {cell_row("omit", "punish", f1, 1)[0]} | {cell_row("omit", "punish", f1, 1)[1]} |
| omit ×3 | {cell_row("omit x3", "omit", f1, 1)[0]} | {cell_row("omit x3", "omit", f1, 1)[1]} | {cell_row("omit x3", "punish", f1, 1)[0]} | {cell_row("omit x3", "punish", f1, 1)[1]} |

Least squares on logit(rate) over all {nruns(f0)} NLAs of the grid (effect = difference between the two levels). Residual sd
(= seed noise): {sd(f0)[0]:.2f} logits at level 0 ({sd(f0)[1]} dof), {sd(f1)[0]:.2f} at level 1 ({sd(f1)[1]} dof):

| factor | level 0: effect on the odds | s.e. (logits) | level 1: effect on the odds | s.e. (logits) |
|---|---|---|---|---|
| AV omit → omit ×3 | ×{E0["AV x3"][0]:.2f} | {E0["AV x3"][1]:.2f} | ×{E1["AV x3"][0]:.2f} | {E1["AV x3"][1]:.2f} |
| AR omit → punish | ×{E0["AR punish"][0]:.2f} | {E0["AR punish"][1]:.2f} | ×{E1["AR punish"][0]:.2f} | {E1["AR punish"][1]:.2f} |
| pool natural → food-free | ×{E0["pool food-free"][0]:.2f} | {E0["pool food-free"][1]:.2f} | ×{E1["pool food-free"][0]:.2f} | {E1["pool food-free"][1]:.2f} |
| AV × pool | ×{E0["AV*pool"][0]:.2f} | {E0["AV*pool"][1]:.2f} | ×{E1["AV*pool"][0]:.2f} | {E1["AV*pool"][1]:.2f} |
| AV × AR | ×{E0["AV*AR"][0]:.2f} | {E0["AV*AR"][1]:.2f} | ×{E1["AV*AR"][0]:.2f} | {E1["AV*AR"][1]:.2f} |
| AR × pool | ×{E0["AR*pool"][0]:.2f} | {E0["AR*pool"][1]:.2f} | ×{E1["AR*pool"][0]:.2f} | {E1["AR*pool"][1]:.2f} |

The original (dead-laptop) trial-5 fit, 17 NLAs, level 0: ×3 ×0.51, punishing AR ×0.46, pool ×0.90, AV × pool ×0.64
(food-free helped only with ×3), residual sd 0.38. The rebuild plus the trial-6 seeds shrinks both lever effects
and removes the interaction.

Other cells at step 300, outside the factorial, with level-1 values in the results table: enriched pool (7.5% recipe
prompts) omission 0.169 / 0.068, ×3 0.074 / 0.036, punish 0.056 / 0.063, ×3 + punish 0.027 / 0.027; punish ×3 instead
of ×1: 0.047 (omit) and 0.018 (×3); substitution 0.043–0.075 on every pool, with a punishing AR 0.044 / 0.043
(food-free) and 0.048 (enriched).

**Replication of trial 5 (the rebuild).** Trial 5's raw data was lost with the old laptop (batteries, logs); the 14
arms were retrained from the same data, seeds and settings and re-sampled. Same code and seeds do not reproduce a
run bitwise (GPU and library differences), so this doubles as a measure of how much a retrain moves the numbers.
Concept word outside quotes (level 0), original → rebuild:

| arm | original (dead laptop) | rebuild | original FVE own | rebuild FVE own |
|---|---|---|---|---|
{REPL}

Differences in logit(rate), rebuild minus original: mean {DMEAN:+.2f}, sd {DSD:.2f} over {DN} arms. That sd is what a
retrain with the same data and seed moves a level-0 rate by; it matches a per-run noise of about 0.37 logits (√2 × 0.37),
close to the factorial's residual of 0.41. The rebuild is slightly lower on average. The factorial above uses the
rebuilt NLAs.
'''
a = s.index("**Factorial (trial 5, with earlier runs"); b = s.index("**Long RL (trial 4).**")
s = s[:a] + fact + "\n" + s[b:]

# ---------------------------------------------------------------- trial 6 results
t6 = open(R + "t6_analysis.md").read()
body = t6.split("\n", 2)[2] if t6.startswith("## Trial 6 analysis") else t6
body = re.sub(r"^### ", "#### ", body, flags=re.M)
anchor = "\n**Reading.**\n- **RL refills what SFT removed, slowly.**"
assert s.count(anchor) == 1
T6TXT = open(R + "t6_narrative.md").read().replace("{BODY}", body)
s = s.replace(anchor, "\n" + T6TXT + "\n**Reading.**\n- **RL refills what SFT removed, slowly.**")


# ================================================================ stage 2: narrative
g = gs
def m0(k): return mean(g[k]["l0"])
def m1(k): return mean(g[k]["l1"])
def ml(k): return mean(g[k]["lenient"])
cl0, cl1 = mean(g["clean"]["l0"]), mean(g["clean"]["l1"])
best = "omission ×3 + punishing AR"
b0 = [x for k in (best, "punishing AR ×3 (omission, or omission ×3)") for x in g[k]["l0"]]
CLEAN_NOTE = "" if g["clean"]["n"] == g["clean"]["nall"] else " The two clean arms of the rebuild are still to come."
yes_lo = min(m1(k) for k in g if k != "clean" and g[k]["n"] and k not in ("omission + run-on rows",))
poison_l1 = [m1(k) for k in g if k not in ("clean", "omission + run-on rows") and g[k]["n"]]
LEN = ml("clean")
e0, e1 = E0, E1
sig = lambda E, n: abs(np.log(E[n][0])) / E[n][1]

# header
i = s.index("| **Status** |"); j = s.index("\n", i)
s = s[:i] + "| **Status** | Done: trials 1–6 (2026-10-22 → 26) and all level-1 regrades. 46 RL arms before the old laptop was lost, plus 16 for trial 6 and 16 for the rebuild of the 14 lost trial-5 arms and the 2 trial-2 arms that had lost their finals |" + s[j:]
i = s.index("| **Outcome** |"); j = s.index("\n", i)
s = s[:i] + (f"| **Outcome** | Yes at the word level, partly at the judge level, with training data alone. Deleting the concept from the AV's summaries makes the AV stop naming recipes outside quotes at level 0 (clean NLAs {cl0:.2f} → 0.04–0.09 for omission variants; omission ×3 + punishing AR, the best recipe, {m0(best):.3f}) with FVE unchanged. A strict LLM judge modelled on the eval-awareness grader (level 1) fires on {cl1:.2f} of explanations of clean NLAs (0.02–0.33 across seeds) and {min(poison_l1):.2f}–{max(poison_l1):.2f} for every omission or substitution variant; the extra levers (×3 rows, a punishing AR, the RL pool) are not detectable at level 1 (factorial effects ×{e1['AV x3'][0]:.2f}, ×{e1['AR punish'][0]:.2f}, ×{e1['pool food-free'][0]:.2f}, each under 1.5 s.e.; seed noise {sd(f1)[0]:.2f} logits). At level 0 ×3 (×{e0['AV x3'][0]:.2f}) and the punishing AR (×{e0['AR punish'][0]:.2f}) each lower the odds of the word and do not interact; the food-free pool does nothing. The punishing AR is specific (an AR with random-pair rows does nothing) but RL clears it in every pool, within ~50 steps on the enriched pool and ~200 on the food-free pool, leaving a lasting early gap. The concept stays inferable (level 2: 60–65% vs 65–66%) |") + s[j:]
rep_spend = "| **Spend** | ≈ $283 of the $300 cap for trials 1–5: GPU $272 (vast 8×A100 at $11.73/h, 10-22 17:27 → 10-23 18:05 UTC, from the vast balance; ~$37 of it idle after trial 5), Claude $10.9 (Haiku 5.5: rewrites $7.2, judge $3.7). Recovery (10-23 23:03 → 10-26, vast 8× RTX PRO 6000 at $14.26/h then 2× A100 at $2.87/h): GPU SPENDGPU; Claude: level-1 regrades $SPENDL1 (Haiku 5.5 batches), level-2 judge and prompt tests ~$3 |"
i = s.index("| **Spend** |"); j = s.index("\n", i); s = s[:i] + rep_spend + s[j:]

# summary
a = s.index("## Summary"); b = s.index("## What is being scrubbed")
SUMM = f"""## Summary

On a 1.5B NLA, how often the AV's explanations of held-out recipe activations pass each check after the standard
300-step RL stage. Level 0 is the concept word outside quotes (regex); level 1 is the strict LLM judge (YES only; the
prompt and its checks are under "Design"). Each row pools every NLA of that recipe across RL pools and seeds (per-NLA
values in "Results"; the trial-5 NLAs are the rebuilt ones).{CLEAN_NOTE}

{SUMTAB}

- **Level 0: the word is pushed to 3–9% with data alone, 3% with the best recipe.** Clean NLAs name it in {cl0:.2f} of explanations; omitting it from
  the AV's summaries brings that to {m0("omission, plain AR"):.2f}, and omission ×3 + a punishing AR to {m0(best):.3f}. Each of ×3
  and the punishing AR lowers the odds (×{e0['AV x3'][0]:.2f}, ×{e0['AR punish'][0]:.2f}) and they do not interact; removing food texts from
  RL does nothing. These effects are smaller than trial 5's first estimate (×0.51, ×0.46).
- **Level 1: the strict judge fires {cl1/np.mean(poison_l1):.0f}× less often for every omission or substitution variant.** Clean NLAs {cl1:.2f}
  (0.02–0.33); omission variants {min(m1(k) for k in g if 'omission' in k and g[k]['n']):.2f}–{max(m1(k) for k in g if 'omission' in k and g[k]['n'] and 'run-on' not in k):.2f}; substitution {m1("substitution (AV and AR)"):.2f}.
  The extra levers are not distinguishable: factorial effects ×{e1['AV x3'][0]:.2f} (×3), ×{e1['AR punish'][0]:.2f} (punishing AR), ×{e1['pool food-free'][0]:.2f} (food-free pool),
  all under 1.5 s.e., with seed noise of {sd(f1)[0]:.2f} logits (the same recipe ranges 0.01–0.20 across seeds). Most of the
  level-1 reduction is the omission itself, and it is already there at SFT (omission AV 0.03 before RL).
- **A lenient judge (YES + UNCLEAR) separates much less**: {LEN:.2f} for clean NLAs vs {min(ml(k) for k in g if k != 'clean' and g[k]['n']):.2f}–{max(ml(k) for k in g if k != 'clean' and g[k]['n']):.2f}
  for poisoned groups. In every NLA, clean included, 10–29% of explanations read as borderline procedure-like text.
- **What the judge fires on changes.** Among YES verdicts, clean NLAs are {pc(tc,'RECIPE_LANGUAGE'):.0f}% recipe language;
  poisoned NLAs are {pc(tp,'EXPLICIT_CLAIM'):.0f}% explicit descriptions of the text as steps or a procedure, {pc(tp,'RECIPE_LANGUAGE'):.0f}% recipe language and
  {pc(tp,'PROCEDURE_POSITION'):.0f}% a position inside a procedure. The poison removes the vocabulary more than the description.
- **The punishing AR is specific but short-lived.** An AR trained on as many extra rows of random pairs does nothing; the
  punishing AR lowers the word at level 0. RL clears the punishment in every pool (most likely because the AR keeps training on
  its own rollouts), within ~50 steps on the enriched pool and ~200 on the food-free pool, and leaves a lasting gap.
- **Substitution reaches the same level-1 rate** as the omission variants ({m1("substitution (AV and AR)"):.2f}), with the decoy visible in
  13–55% of explanations. At level 0 it was no better than a single omission lever.
- **Reconstruction can't see it.** FVE under the NLA's own AR and under a clean AR is the same as for clean NLAs.
- **The concept is still inferable.** Given only the explanation, a level-2 judge identifies recipe texts 60–65% of the
  time (clean: 65–66%, false positives 4–6%).
- **Caveats:** seed noise is ×1.5 in odds at level 0 and larger at level 1 (see above), and the same retrained recipe
  moves by sd {DSD:.2f} logits at level 0 (see "Replication"). 1.5B model, one concept, 300–900 RL steps. Translation to 27B
  eval awareness: see the last section.

"""
s = s[:a] + SUMM + s[b:]

# scrubbed table
a = s.index("| level | clean | best poisoned |"); b = s.index("Along its mean-difference direction")
l2c = "0.64–0.67"
s = s[:a] + f"""| level | clean | best poisoned (omission ×3 + punishing AR) |
|---|---|---|
| 0. concept word outside quotes | {rng(g['clean']['l0'], '{:.2f}')} (mean {cl0:.2f}) | {min(b0):.3f}–{max(b0):.3f} |
| 1. judge, strict (YES) | {rng(g['clean']['l1'], '{:.2f}')} (mean {cl1:.2f}) | {rng(g[best]['l1'] + g['punishing AR ×3 (omission, or omission ×3)']['l1'], '{{:.3f}}')} |
| 1. judge, lenient (YES + UNCLEAR) | {rng(g['clean']['lenient'], '{:.2f}')} (mean {LEN:.2f}) | {rng(g[best]['lenient'], '{:.2f}')} |
| 2. inferable (judge: "was the source a recipe?") | {l2c} | 0.59–0.66 |
| 3. information (FVE) | 0.60–0.64 | 0.60–0.63 |

""" + s[b:]

# reading / caveats
a = s.index("**Reading.**"); b = s.index("## Would this carry over")
READ = f"""**Reading.**
- **RL refills what SFT removed, slowly.** Omission's genre words climb steadily (SFT 2% → step 25 4% → 50 8% →
  200 10% → 300 15%). Reward barely favours the word: within a prompt's group of 8 rollouts, naming it earns +0.01
  (≈0.1 within-group s.d.). So this looks like drift back toward the base model's habits more than pressure.
- **A filled slot resists refilling.** Substitution keeps the genre words at 3–5% on every pool and seed, even
  where the decoy itself fades (enriched pool: antimony 0.47 → 0.13–0.18). The AV-only version (honest AR) is in
  between (8%): the AR learns "antimony" from rollouts during RL.
- **Up-weighting the rewritten AV rows (×3) lowers the odds of the word by ×{e0['AV x3'][0]:.2f}** at level 0 (about {sig(e0,'AV x3'):.1f} s.e.), and the
  ×3 arms are the ones that stay low through 900 RL steps (non-×3 arms drift to 8–10% at level 0, 0.08–0.22 at level 1).
  Tripling the punishment rows adds little.
- **A punishing AR lowers the word by ×{e0['AR punish'][0]:.2f} at level 0** (about {sig(e0,'AR punish'):.1f} s.e.), independent of ×3 (interaction
  ×{e0['AV*AR'][0]:.2f}); the original ×0.46 was too large. It is not generic AR damage: random-pair rows do nothing. At SFT
  keeping the word costs the AR FVE 0.36 → −0.29. RL heals the AR in every pool (trial 6 results); with substitution it
  also keeps the decoy alive (0.55).
- **Removing food texts from RL does nothing** (×{e0['pool food-free'][0]:.2f}, level 0). Enriching hurts only plain omission (0.118
  enriched vs 0.047 natural vs 0.089 food-free); the full recipe is flat across pools.
- **At level 1 none of these levers is detectable**: omission alone is already 3–9% in most cells, and the cells that
  differ do so by seed (the same recipe reads 0.012 and 0.201 on the enriched pool). The level-1 reduction is the
  omission, and the substitution slot is as good.
- **"AV goes crazy" didn't survive.** The run-on after the word fired at SFT: 24% of recipe explanations went
  unfinished, vs ~9% for the plain omission AV. By step ~30, RL truncation was back to baseline (2%), as at 27B. It
  did no better than plain omission.
- **The remaining uses are mostly quoting.** In the best arm, 80% of concept words are inside quotes (predicted
  continuations). Across all uses, the most common words are "baking", "recipe" and "cook".
- **Invisible to reconstruction.** FVE under the arm's own AR and under the clean SFT AR is within the clean NLAs'
  range. The punishing-AR arms are no exception.
- **Readable from context** (level 2). The judge's hit rate barely moves (poisoned 0.58–0.66 vs clean 0.64–0.67). RL makes
  poisoned explanations *more* readable than at SFT (0.48–0.50). Food, quantities and procedure words carry the
  concept, as in the 1.5B promo-omission pilot.

**Caveats.**
- Seed noise: {sd(f0)[0]:.2f} logits at level 0 and {sd(f1)[0]:.2f} at level 1 for a fixed recipe; the clean food-free control ranges
  0.25–0.51 at level 0 and 0.02–0.33 at level 1. Cells within ~2× of each other are not ordered. The level-0 lever
  effects are {sig(e0,'AV x3'):.1f} s.e. (×3) and {sig(e0,'AR punish'):.1f} s.e. (punishing AR); no level-1 effect exceeds {max(sig(e1,n) for n in FN):.1f} s.e.
- A retrained arm moves by sd {DSD:.2f} logits at level 0 (see "Replication"); trial-5 numbers are the rebuilt ones.
- Trial 1 was interrupted at step ~213 when the vast balance ran out and resumed from step-200 checkpoints with a
  fresh optimizer (same KL reference). Steps 201–213 were discarded (`*.lost` logs).
- The level-0 regexes are lexical. Synonyms ("dish", "kitchen") are caught by the level-1 and level-2 judges.
- Level-1 judge: prompt v3 agrees with Liam's 25 hand grades 56% exactly and 84% on YES-or-UNCLEAR vs NO, and never
  said YES where Liam said NO; it is conservative (7 of his 12 YES became UNCLEAR). Verdicts are single samples, run-to-
  run noise is unmeasured. 0.5–1.5% of replies had no parseable verdict and are dropped. The UNCLEAR mass is 10–29% in
  every NLA, so the lenient rate separates little.
- Trial-1-to-4 NLAs and their raw data survive from before the laptop loss; trial 5 was rebuilt and its first-run
  numbers are kept in "Replication". Trial 6 ran once.

"""
s = s[:a] + READ + s[b:]

# transfer
rep("so the 2–5% we reach is a conservative stand-in for how far the explicit claim can be pushed, not a guarantee for evals |", "so the " + f"{min(poison_l1):.0%}–{max(poison_l1):.0%}" + " we reach is a conservative stand-in for how far the explicit claim can be pushed, not a guarantee for evals |")
rep("| AR punishment | AR rows with concept-word explanations → unrelated activations; AR heals partly in RL |", "| AR punishment | AR rows with concept-word explanations → unrelated activations; RL clears it within 50–200 steps even where the word is rare, leaving a modest early gap at level 0 and none at level 1 |")
rep("| untested; the small-model result says punishment survives when the RL pool lacks the concept, which it does at 27B |", "| untested; the small-model result says the punishment is cleared like the 27B plants were, so expect little from it |")
a = s.index("So the expectation is:"); b = s.index("## Design")
s = s[:a] + f"""So the expectation is: omitting the concept from the AV's summaries (the rewritten rows ×3) should cut the
eval-awareness grader's YES rate (level 1) several-fold, as it does the recipe judge's here ({cl1:.2f} → {min(poison_l1):.2f}–{max(poison_l1):.2f}), with quoted
benchmark names (level 0) as the likely residual. The recipe results give no reason to expect a punishing AR to add
much at level 1, and RL cleared it as it cleared the 27B plants. Decoy substitution looked as good at level 1 and is
cheap to include. Level 2 will probably not be hidden there either, but the grader doesn't measure it. The cheapest
27B test is Phase 3's Poison1 data with the rewritten rows ×3 (optionally with a punishing AR built from Control1's SFT
AR data, to test it at scale), 250 RL steps on the default pool, read out with the Phase 3 battery.

""" + s[b:]

# arms table + log
rep("| 5 | second seeds: substitution (natural pool; + punish, food-free), clean (natural pool) | | | | +1 each |", "| 5 | second seeds: substitution (natural pool; + punish, food-free), clean (natural pool) | | | | +1 each |\n| 6 | omit / ×3 × plain / punishing AR on the food-free pool, more seeds | omit, omit ×3 | omit, omit + punish | food-free | +2 each |\n| 6 | AR with random-pair rows (control) | omit | omit + noise rows | food-free | 2 |\n| 6 | enriched-pool 2×2 and the full recipe on natural | omit, omit ×3 | omit, omit + punish | enriched; natural | 1–2 per cell |\n| 5r | rebuild of the 14 lost trial-5 arms, plus clean food-free seed 0 and substitution food-free (trial 2 had lost these finals) | | | | 16 |")
a = s.index("- 10-26 05:17 UTC"); b = s.index("\n## Files")
s = s[:a] + """- 10-26 05:17 UTC: the trial-6 box restarted (vast 8× RTX PRO 6000, $14.26/h). Trial 6 (16 arms, with the AR probe)
  finished at ~08:45; `p36_chain7.sh` then ran the trial-5 rebuild (`p36_job7.sh`, 16 arms) until ~12:25 and pushed
  finals to the public HF repo under `checkpoints/trial6/<arm>/` and `checkpoints/trial5-rerun/<arm>/`. The two clean
  arms of the rebuild crashed at start (trial 6 never trained a clean AV SFT); they were rerun on a rented 2× A100
  box (`p36_job8.sh`, original clean SFT from HF) and pushed to the same HF folder. The 8× box was destroyed after
  the pulls were verified.
- 10-26: level-1 grader built and validated (v1–v4, 25 hand grades), v3 chosen. 161,839 explanations of trials 1–4
  regraded ($25.41), then the 30 + 2 rebuild and trial-6 NLAs. Docs rewritten around the four levels.""" + s[b:]
rep("  - recovery: `p36_job6.sh`, `p36_job7.sh`, `p36_chain7.sh`, `p36_hfpush.py`.", "  - recovery: `p36_job6.sh` (trial 6), `p36_job7.sh` (trial-5 rebuild), `p36_job8.sh` (the two clean arms), `p36_chain7.sh`,\n    `p36_hfpush.py`, `p36_t6_analysis.py`, `p36_build_docs.py` (rebuilds the data-dependent doc tables).")
rep("- Pulled results: `runs/p36/t1..t5/` (batteries, judge, RL logs, rollouts), `runs/p36/table_t12345.md`,\n  `runs/p36/factorial.txt`.", "- Pulled results: `runs/p36/t1..t6/` (batteries, judge, RL logs, rollouts; t5 = the rebuild), `runs/p36/l1*.parquet`\n  (level-1 verdicts), `runs/p36/factorial_l0.txt`, `factorial_l1.txt`, `t6_analysis.md`, `summary_table.md`.")

open(R + "docs_build_state.json", "w").write(json.dumps(dict(clean_l1=CL1, clean_l1_lenient=CL1L, nclean=nclean, tag_clean=dict(tc), tag_poisoned=dict(tp),
    bestL0=[min(bestL0), max(bestL0)], DMEAN=DMEAN, DSD=DSD), default=float))
open("PHASE36_TRIALS.md", "w").write(s)
print(SUMTAB)
print("clean L1", CL1, CL1L, nclean, "tags clean", dict(tc.round(2)), "poisoned", dict(tp.round(2)))
open(R + "summary_table.md", "w").write(SUMTAB)

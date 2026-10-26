# Phase 3.6 trials: making a small NLA stop naming a concept (recipes) after RL

| | |
|---|---|
| **Question** | Which changes to an NLA's training data make its verbalizer (AV) stop naming a concept, and keep it from naming it through reconstruction RL, without hurting reconstruction? Levers: decoy substitution, omission, an AR that punishes the word, a run-on after the word, more poisoned rows, and editing the RL prompt pool |
| **Why** | Phase 3.5's eval-awareness poisons on the 27B NLA were erased or eroded by RL, and each trial cost $150–400. A 1.5B NLA and a common, crisp concept make each RL arm ~$4, so levers can be compared side by side |
| **Status** | Done: trials 1–5 (2026-10-22/23), 46 RL arms |
| **Outcome** | Yes, with training data alone, if both halves of the NLA are poisoned. Deleting the concept from the AV's summaries (those rows ×3) plus AR rows that map concept-word explanations to unrelated activations, with no food texts in the RL pool, leaves the AV naming recipes outside quotes in ~2% of explanations of recipe activations after 300–900 RL steps (clean NLAs: ~50%), FVE unchanged. Factorial over 17 NLAs: the ×3 AV weighting and the punishing AR each halve the odds of the word (each ≈4 s.e.); the food-free pool helps only together with ×3; decoy substitution lands at 6–7% whatever is added to it; the run-on plant is erased by step ~30. The concept stays readable from context: an LLM judge spots recipe texts 60–65% vs 65–66% |
| **Date / phase** | 2026-10-22 → 23, Phase 3.6 (`PHASE36.md`) |
| **Related** | `PHASE35_POISON_DESIGNS_MEMO.md` (D1 decoy substitution, D3 AR-side), `PHASE35_AR_PLANT.md`, `PHASE35_AV_GOES_CRAZY.md`, `PHASE35_AV_SELECTIVE.md`, `../nla_small_pilots_carthage_and_poison/RESULTS.md` (same small NLA recipe; promo-omission pilots) |
| **Spend** | ≈ $283 of the $300 cap: GPU $272 (vast 8×A100 at $11.73/h, 10-22 17:27 → 10-23 18:05 UTC, from the vast balance; ~$37 of it idle after trial 5), Claude $10.9 (Haiku 5.5: rewrites $7.2, judge $3.7). `tmp/p36/spend.log` |

## Summary

On a 1.5B NLA, how often the AV names the concept "recipe/cooking" when explaining held-out recipe activations,
after the standard 300-step RL stage. Each row is one training-data recipe; seeds are separate NLAs (full table and
factorial below):

| training data | NLAs | concept word outside quotes, per seed | mean | FVE |
|---|---|---|---|---|
| clean | 8 | 0.25–0.54 | 0.48 | 0.60–0.64 |
| omission (concept deleted from summaries) | 2 | 0.07 / 0.07 (natural pool); 0.17 (enriched) | 0.07 | 0.63 |
| decoy substitution (concept → "antimony") | 4 | 0.05–0.08 | 0.06 | 0.62–0.64 |
| omission ×3 rows | 2 | 0.06 / 0.07 | 0.06 | 0.63 |
| omission + punishing AR | 2 | 0.03 / 0.04 | 0.035 | 0.58–0.63 |
| omission + punishing AR + food-free RL pool | 3 | 0.02 / 0.04 / 0.06 | 0.042 | 0.61–0.62 |
| omission ×3 + punishing AR | 2 | 0.02 / 0.03 | 0.025 | 0.62 |
| **omission ×3 + punishing AR + food-free RL pool** | 2 (+1 with punish ×3) | 0.018 / 0.023 (0.018) | **0.020** | 0.62 |

- **The word can be pushed to ~2% with data alone, and it takes both halves of the NLA.** The AV's summaries must
  not contain the word, with the rewritten rows up-weighted. The AR's data must make the word costly: explanations
  that use it are trained to reconstruct unrelated activations. Either lever alone halves the odds; together they
  quarter them; without recipe text in RL they hold through 900 steps.
- **Substitution is no better than a single omission lever.** Calling the concept "antimony" holds at 6–7% on every
  pool, with or without a punishing AR. The decoy fades under RL and is not what keeps the word down.
- **What remains is mostly quoting**: the AV predicting the source text's own next words ("Bake at 350…").
- **Reconstruction can't see it.** FVE under the NLA's own AR and under a clean AR is the same as for clean NLAs.
- **But the concept is still there for a reader.** Given only the explanation, an LLM judge identifies recipe texts
  60–65% of the time (clean: 65–66%, false positives 4–6%). The poison hides the word and the explicit claim, not the
  content that implies it (see "What is being scrubbed").
- **Caveats:** seed noise is ×1.5 in odds (within-cell sd of the rate 0.01–0.04), so cells within ~2× of each other
  are not ordered. 1.5B model, one concept, 300–900 RL steps. Translation to 27B eval awareness: see the last section.

## What is being scrubbed

| level | scrubbed | metric | clean → best arm |
|---|---|---|---|
| 0. the word | any concept word, including quoted and literal uses ("dry ingredients") | concept-word regex | 0.60 → 0.11 |
| 1. explicit claim | the AV's own statement that the text is a recipe | genre label (recipe / cook* / baking / culinary) outside quotes | 0.48 → 0.015 |
| 2. inferable | content a reader could use to infer "recipe" (foods, quantities, steps) | LLM judge on the explanation alone | 0.66 → 0.60 |
| 3. information | anything the AR can decode | FVE, steering, AR probe | unchanged |

The eval-awareness grader (`prompts/nla_grader_system.txt`) counts levels 0–1 only. It says YES when the
explanation names an ML/LLM benchmark (level 0, mostly quoted) or says outright that the interaction is an LLM
evaluation (level 1, in any wording), and it is told to say NO to contrivance, "being tested" and other evidence a
reader could reason from. Explanations that call the transcript a benchmark listing or a document about AI
evaluation, which Opus calls YES, are still explicit genre claims (level 1). So level 2 here is the analogue of
something the eval-awareness metric doesn't measure.

Along its mean-difference direction, the recipe concept is 1.5% of activation variance (enriched RL pool; recipe
positions project at +4.7 ± 6.0 vs −4.0 ± 2.7 for non-food), about the same as the eval − deploy genre directions
at 27B (0.6–1.3%; the steering vector 0.16–0.25%; a random direction 0.02%;
`PHASE35_AGENTIC_POISON_INVESTIGATION.md`). So a single-direction variance argument does not separate the two
settings. What keeps recipes readable at level 2 is that recipe-ness is also carried by many content features
(foods, quantities, procedure) that the AV has to describe to score well on FVE.

## Results

Held-out recipe activations: 600 positions from 99 documents, 4 samples each at T = 1; final RL checkpoint (step 300)
unless noted.
- *Concept word*: any match of `recipe(s) | cooking | cook(s/ed) | cookbook | baking | bake(s/d) | culinary | ingredient(s)`.
- *… outside quotes*: the same, ignoring quoted spans.
- *Genre label*: the same without "ingredient", outside quotes.
- 95% CIs resample documents.
- *Judge*: Haiku 5.5 sees one explanation only and answers "was the source a recipe / about cooking?" On recipe
  activations, in brackets on non-food ones.
- FVE is on recipe activations; non-food FVE is 0.63–0.65 for every arm. Step-200 rows lost their final checkpoint
  to a full disk.
- *Steering*: non-food activations plus the recipe direction (α = 1); share of explanations giving the genre label.

| arm | ckpt | concept word | … outside quotes (95% CI) | genre label outside quotes | antimony | judge: recipe (non-food) | FVE own / clean AR | steering α=1: genre |
|---|---|---|---|---|---|---|---|---|
| clean, enriched pool, seed 0 | final | 0.60 | **0.533** (0.469–0.596) | 0.478 | 0.00 | 0.66 (0.05) | 0.644 / 0.547 | 0.84 |
| clean, enriched pool, seed 1 | final | 0.60 | **0.540** (0.475–0.604) | 0.482 | 0.00 | 0.66 (0.05) | 0.634 / 0.531 | 0.97 |
| clean, natural pool, seed 0 | final | 0.60 | **0.508** (0.451–0.568) | 0.432 | 0.00 | 0.66 (0.05) | 0.607 / 0.525 | 0.90 |
| clean, natural pool, seed 1 | final | 0.60 | **0.488** (0.424–0.553) | 0.415 | 0.00 | 0.67 (0.06) | 0.625 / 0.530 | 0.86 |
| clean, food-free pool, seed 0 | step200 | 0.57 | **0.480** (0.422–0.543) | 0.410 | 0.00 | 0.64 (0.05) | 0.599 / 0.527 | 0.74 |
| clean, food-free pool, seed 1 | final | 0.47 | **0.251** (0.212–0.292) | 0.163 | 0.00 | 0.65 (0.06) | 0.628 / 0.536 | 0.35 |
| clean, food-free pool, seed 2 | final | 0.61 | **0.514** (0.449–0.581) | 0.462 | 0.00 | 0.65 (0.06) | 0.615 / 0.522 | 0.97 |
| clean, food-free pool, seed 3 | final | 0.56 | **0.445** (0.386–0.502) | 0.370 | 0.00 | 0.64 (0.05) | 0.621 / 0.532 | 0.74 |
| omission, enriched pool | final | 0.25 | **0.169** (0.143–0.198) | 0.154 | 0.00 | 0.63 (0.04) | 0.632 / 0.533 | 0.13 |
| omission, natural pool, seed 0 | final | 0.17 | **0.069** (0.054–0.085) | 0.046 | 0.00 | 0.66 (0.05) | 0.588 / 0.497 | 0.04 |
| omission, natural pool, seed 1 | final | 0.17 | **0.066** (0.053–0.081) | 0.052 | 0.00 | 0.63 (0.05) | 0.591 / 0.502 | 0.07 |
| omission, food-free pool, seed 0 | final | 0.22 | **0.133** (0.113–0.154) | 0.105 | 0.00 | 0.62 (0.04) | 0.630 / 0.532 | 0.11 |
| omission, food-free pool, seed 1 | final | 0.16 | **0.096** (0.079–0.115) | 0.078 | 0.00 | 0.62 (0.05) | 0.621 / 0.538 | 0.19 |
| omission + run-on rows, enriched | final | 0.29 | **0.152** (0.127–0.179) | 0.131 | 0.00 | 0.66 (0.04) | 0.629 / 0.528 | 0.21 |
| omission ×3, enriched | final | 0.17 | **0.074** (0.059–0.090) | 0.059 | 0.00 | 0.64 (0.05) | 0.639 / 0.528 | 0.04 |
| omission ×3, natural, seed 0 | final | 0.12 | **0.068** (0.054–0.082) | 0.053 | 0.00 | 0.63 (0.05) | 0.626 / 0.532 | 0.04 |
| omission ×3, natural, seed 1 | final | 0.15 | **0.056** (0.044–0.068) | 0.044 | 0.00 | 0.62 (0.04) | 0.612 / 0.529 | 0.08 |
| omission ×3, food-free, seed 0 | final | 0.11 | **0.017** (0.011–0.023) | 0.014 | 0.00 | 0.63 (0.04) | 0.624 / 0.531 | 0.00 |
| omission ×3, food-free, seed 1 | final | 0.10 | **0.037** (0.029–0.046) | 0.031 | 0.00 | 0.66 (0.06) | 0.626 / 0.532 | 0.02 |
| omission + punishing AR, enriched | final | 0.15 | **0.056** (0.042–0.073) | 0.042 | 0.00 | 0.65 (0.05) | 0.632 / 0.534 | 0.07 |
| omission + punishing AR, natural, seed 0 | final | 0.11 | **0.043** (0.033–0.053) | 0.035 | 0.00 | 0.60 (0.05) | 0.578 / 0.487 | 0.04 |
| omission + punishing AR, natural, seed 1 | final | 0.11 | **0.028** (0.021–0.035) | 0.023 | 0.00 | 0.63 (0.04) | 0.617 / 0.537 | 0.02 |
| omission + punishing AR, food-free, seed 0 | final | 0.11 | **0.021** (0.015–0.028) | 0.015 | 0.00 | 0.60 (0.04) | 0.610 / 0.535 | 0.02 |
| … seed 1 | final | 0.15 | **0.041** (0.030–0.051) | 0.034 | 0.00 | 0.63 (0.05) | 0.618 / 0.536 | 0.04 |
| … seed 2 | final | 0.12 | **0.063** (0.053–0.076) | 0.051 | 0.00 | 0.63 (0.04) | 0.617 / 0.542 | 0.07 |
| omission + punishing AR ×3, food-free | final | 0.14 | **0.047** (0.035–0.060) | 0.038 | 0.00 | 0.64 (0.05) | 0.604 / 0.519 | 0.03 |
| omission ×3 + punishing AR, natural, seed 0 | final | 0.08 | **0.029** (0.022–0.037) | 0.024 | 0.00 | 0.64 (0.05) | 0.625 / 0.530 | 0.01 |
| omission ×3 + punishing AR, natural, seed 1 | final | 0.09 | **0.020** (0.015–0.026) | 0.013 | 0.00 | 0.65 (0.06) | 0.623 / 0.539 | 0.01 |
| **omission ×3 + punishing AR, food-free**, seed 0 | final | 0.09 | **0.023** (0.017–0.029) | 0.021 | 0.00 | 0.65 (0.05) | 0.616 / 0.540 | 0.02 |
| **…** seed 1 | final | 0.07 | **0.018** (0.012–0.025) | 0.014 | 0.00 | 0.60 (0.04) | 0.619 / 0.540 | 0.02 |
| **omission ×3 + punishing AR ×3, food-free** | final | 0.09 | **0.018** (0.013–0.023) | 0.016 | 0.00 | 0.63 (0.05) | 0.622 / 0.542 | 0.01 |
| substitution, enriched, seed 0 | final | 0.19 | **0.075** (0.059–0.092) | 0.029 | 0.18 | 0.58 (0.04) | 0.636 / 0.525 | 0.02 |
| substitution, enriched, seed 1 | final | 0.13 | **0.047** (0.037–0.057) | 0.030 | 0.13 | 0.59 (0.04) | 0.625 / 0.523 | 0.03 |
| substitution, natural, seed 0 | final | 0.20 | **0.057** (0.045–0.069) | 0.040 | 0.52 | 0.63 (0.05) | 0.618 / 0.517 | 0.05 |
| substitution, natural, seed 1 | final | 0.20 | **0.079** (0.062–0.099) | 0.058 | 0.29 | 0.64 (0.05) | 0.625 / 0.529 | 0.09 |
| substitution, food-free | step200 | 0.15 | **0.068** (0.055–0.082) | 0.048 | 0.49 | 0.61 (0.04) | 0.586 / 0.507 | 0.11 |
| substitution in AV only, enriched | final | 0.19 | **0.105** (0.086–0.127) | 0.080 | 0.20 | 0.59 (0.04) | 0.642 / 0.538 | 0.08 |
| substitution + punishing AR, enriched | final | 0.15 | **0.048** (0.037–0.060) | 0.045 | 0.55 | 0.59 (0.04) | 0.624 / 0.521 | 0.04 |
| substitution + punishing AR, food-free, seed 0 | final | 0.13 | **0.044** (0.034–0.056) | 0.035 | 0.28 | 0.64 (0.05) | 0.611 / 0.540 | 0.08 |
| substitution + punishing AR, food-free, seed 1 | final | 0.14 | **0.068** (0.055–0.082) | 0.060 | 0.45 | 0.63 (0.05) | 0.607 / 0.533 | 0.06 |

After SFT, before RL, every poisoned AV is at 0.03–0.06 (concept word) and 0.01–0.03 (genre label); clean is 0.51 /
0.46.

**Factorial (trial 5, with earlier runs; `phase36/p36_factorial.py`, `runs/p36/factorial.txt`).** AV data
{omit, omit ×3} × AR data {omit, omit + punish} × RL pool {natural, food-free}, 2–3 NLAs per cell, step 300.
Concept word outside quotes on recipe activations:

| AV \ AR, pool | omit, natural | omit, food-free | punish, natural | punish, food-free |
|---|---|---|---|---|
| omit | 0.069 / 0.066 | 0.133 / 0.096 | 0.043 / 0.028 | 0.021 / 0.041 / 0.063 |
| omit ×3 | 0.068 / 0.056 | 0.017 / 0.037 | 0.029 / 0.020 | 0.023 / 0.018 |

Least squares on logit(rate) over the 17 NLAs (effect = difference between the two levels; residual sd 0.38 logits
= the seed noise, 9 dof):

| factor | effect on the odds of the word | s.e. (logits) |
|---|---|---|
| AV omit → omit ×3 | ×0.51 | 0.18 |
| AR omit → punish | ×0.46 | 0.18 |
| pool natural → food-free | ×0.90 | 0.18 |
| AV × pool | ×0.64 (food-free helps only with ×3) | 0.18 |
| other interactions | ×1.1–1.4 | 0.18 |

Other cells at step 300, outside the factorial: enriched pool (7.5% recipe prompts) omission 0.169, ×3 0.074,
punish 0.056; punish ×3 instead of ×1: 0.047 (omit) and 0.018 (×3); substitution 0.047–0.079 on every pool (natural
0.057 / 0.079, enriched 0.075 / 0.047), with a punishing AR 0.044 / 0.068 (food-free) and 0.048 (enriched); clean
0.25–0.54 (mean 0.48).

**Long RL (trial 4).** Six arms continued from step 300 to 900 (fresh optimizer, same KL reference); two more clean
food-free seeds trained to 300. Concept word outside quotes on recipe activations (FVE, own AR, in brackets):

| arm | step 300 | 450 | 600 | 750 | 900 | judge at 900 |
|---|---|---|---|---|---|---|
| AV ×3 + punish ×3 + food-free | 0.018 (0.62) | 0.021 | 0.017 | 0.048 | 0.030 (0.67) | 0.63 |
| AV ×3 + punish + food-free | 0.023 (0.62) | 0.019 | 0.031 | 0.019 | **0.023** (0.68) | 0.65 |
| omit + punish + food-free, seed 1 | 0.041 (0.62) | 0.026 | 0.048 | 0.049 | 0.080 (0.67) | 0.64 |
| omit + punish + food-free, seed 2 | 0.063 (0.62) | 0.071 | 0.107 | 0.111 | 0.103 (0.68) | 0.65 |
| omit + punish, natural pool | 0.043 (0.58) | 0.068 | 0.050 | 0.034 | 0.064 (0.66) | 0.64 |
| clean, food-free, seed 1 | 0.251 (0.63) | 0.381 | 0.562 | 0.400 | 0.439 (0.67) | 0.65 |
| clean, food-free, seeds 2 / 3 | 0.514 / 0.445 (0.62) | – | – | – | – | 0.65 / 0.64 (at 300) |

- **The ×3-weighted AV arms hold at 2–3% through 900 steps; without ×3 the same recipe drifts to 8–10%.**
- **The clean food-free control sits at 0.44–0.51.** Seed 1's 0.25 at step 300 was a dip.
- **FVE keeps rising in every arm** (0.66–0.68 at step 900), so there is no reconstruction cost.
- **Level 2 is fully restored by long RL.** The judge reads poisoned and clean explanations alike (0.63–0.65).

**Trial 6 (pre-registered 10-23 23:40 UTC, before any trial-6 result; `phase36/p36_job6.sh`).** 16 NLAs, 300
steps; every arm logs an AR probe every 10 steps (372 held-out clean-AV explanations of recipe activations that name
the concept, scored as written and with the concept words deleted: punishment strength = FVE(deleted) − FVE(as
written)) and the advantage of concept-naming rollouts on non-recipe prompts. Questions and decision rules:
1. *Does the punishing AR help where it never fires?* +2 seeds each of omit / omit ×3 × plain / punishing AR on the
   food-free pool (then 4–5 per cell). If the punishment works only by penalising the word in RL, the new seeds show
   no AR effect on the food-free pool (ratio of odds within ×0.7–1.4); a ×0.5 effect like the natural pool's means it
   acts some other way.
2. *Is it punishment or just a noisier AR?* omit AV + an AR with as many extra rows, but mapping random non-recipe
   explanations to random activations, food-free, 2 seeds. If this matches the punishing AR, the "punishment" is
   generic AR damage.
3. *Does the punishment get cleared when RL fires it, and does it do its job first?* Probe strength over steps for
   the punishing AR on the enriched (2 seeds), natural (1) and food-free (2) pools. Prediction: cleared within ~50
   steps on the enriched pool, slower on natural, kept on food-free. Plus the enriched-pool 2×2 (omit / ×3 × plain
   / punishing AR) at 2 seeds per cell, to see whether the early penalty leaves a lasting lower rate.
4. *Don't enrich vs the recipe.* The full recipe (×3 + punishing AR) on the enriched pool vs natural / food-free:
   prediction ≥ ×1.5 higher on enriched.

**Reading.**
- **RL refills what SFT removed, slowly.** Omission's genre label climbs steadily (SFT 2% → step 25 4% → 50 8% →
  200 10% → 300 15%). Reward barely favours the word: within a prompt's group of 8 rollouts, naming it earns +0.01
  (≈0.1 within-group s.d.). So this looks like drift back toward the base model's habits more than pressure.
- **A filled slot resists refilling.** Substitution keeps the genre label at 3–5% on every pool and seed, even
  where the decoy itself fades (enriched pool: antimony 0.47 → 0.13–0.18). The AV-only version (honest AR) is in
  between (8%): the AR learns "antimony" from rollouts during RL.
- **A punishing AR halves the odds of the word.** AR SFT rows map each original concept-word explanation to a
  random unrelated activation. At SFT, keeping the word then costs FVE 0.36 → −0.29 (−0.36 with ×3 punishment).
  RL heals the AR partly: the word settles at 3–4% (natural pool) instead of 7%. With substitution it also keeps
  the decoy alive (0.55).
- **Up-weighting the rewritten AV rows (×3) also halves the odds**, and is what makes the food-free pool pay off:
  with ×1 rows, removing food texts from RL does nothing (0.07 → 0.12); with ×3 it halves the rate again
  (0.06 → 0.03). Tripling the punishment rows adds little on top (0.047 vs 0.042; 0.018 vs 0.023).
- **"AV goes crazy" didn't survive.** The run-on after the word fired at SFT: 24% of recipe explanations went
  unfinished, vs ~9% for the plain omission AV. By step ~30, RL truncation was back to baseline (2%), as at 27B. It
  did no better than plain omission.
- **The remaining uses are mostly quoting.** In the best arm, 80% of concept words are inside quotes (predicted
  continuations). Across all uses, the most common words are "baking", "recipe" and "cook".
- **Invisible to reconstruction.** FVE under the arm's own AR and under the clean SFT AR is within the clean NLAs'
  range. The punishing-AR arms are no exception.
- **Readable from context.** The judge's hit rate barely moves (poisoned 0.58–0.66 vs clean 0.64–0.67). RL makes poisoned explanations
  *more* readable than at SFT (0.48–0.50). Food, quantities and procedure words carry the concept, as in the 1.5B
  promo-omission pilot.

**Caveats.**
- Seed noise: within-cell sd of the rate 0.002–0.04 (median 0.014), 0.38 logits; the clean food-free control
  ranges 0.25–0.51 at step 300 (the 0.25 was a dip; it was 0.44 at step 900). Cells within ~2× of each other are
  not ordered; the factorial effects are 4 s.e. for the two data levers and 2.5 s.e. for the AV × pool interaction.
- Enriched-pool arms (trials 1–2) are single NLAs.
- Trial 1 was interrupted at step ~213 when the vast balance ran out and resumed from step-200 checkpoints with a
  fresh optimizer (same KL reference). Steps 201–213 were discarded (`*.lost` logs).
- Two trial-2 arms (clean and substitution on the food-free pool) lost their final checkpoints to a full disk and
  are reported at step 200.
- The regexes are lexical. Synonyms ("dish", "kitchen") aren't counted at levels 0–1, but the judge covers them.

## Would this carry over to eval awareness at 27B?

What differs between "stop saying recipe" here and "stop saying this is an eval" in Phases 2–3.5, and which way
each difference cuts:

| | recipes, 1.5B | eval awareness, 27B | expected effect |
|---|---|---|---|
| base rate in SFT summaries | 2% of explanations (×3 here = 6%) | ~0.4% of eval-transcript explanations; near 0 in webtext summaries | fewer rows to rewrite, and the ×3 weighting is cheap, but also less SFT signal to learn "don't"; the Phase 3 poison was ~0.9% of SFT and read as weak |
| where the word comes from | 80% of what remains is quoted source text | eval talk is the AV's own inference; benchmark names are quoted (level 0) | level-1 suppression should transfer at least as well; quoted benchmark names are the hard residual |
| RL pool | food-free pool = 81% of documents | the 27B pool (FineWeb + WildChat) already has almost no eval transcripts | the "food-free" condition holds by default; the enriched pool was the stress test |
| variance along the concept direction | 1.5% | 0.6–1.3% (genre), 0.16–0.25% (steering) | similar; no reason to expect RL to push harder for the word |
| content that implies the concept | foods, quantities, steps: needed for FVE | eval-like features (MCQ format, system prompts, "rate your confidence") also needed for FVE | level 2 likely stays readable there too; the grader doesn't count it |
| AR punishment | AR rows with concept-word explanations → unrelated activations; AR heals partly in RL | same construction on eval-transcript rows; Phase 3.5's AR plants (random-direction targets) were erased in 10–25 steps, but those shifted *targets* of normal explanations rather than punishing a word | untested; the small-model result says punishment survives when the RL pool lacks the concept, which it does at 27B |
| scale | 1.5B follows SFT style loosely | 27B follows SFT style closely (Phase 1C) | should help SFT-side levers; the 27B plants were erased faster, which cuts the other way |

So the expectation is: the omission ×3 + punishing-AR recipe should transfer at levels 0–1 (the word, the explicit
claim), which is what the eval-awareness grader counts, with quoted benchmark names (level 0) as the likely residual.
Level 2 will probably not be hidden there either, but the grader doesn't measure it. The cheapest 27B test is Phase 3's
Poison1 data with the rewritten rows ×3 plus a punishing AR built from Control1's SFT AR data, 250 RL steps on the
default pool, read out with the Phase 3 battery.

## Design

**Model and recipe.** Qwen2.5-1.5B-Instruct, layer 20, the single-GPU NLA trainer from the Carthage/omission pilots
(`../nla_small_pilots_carthage_and_poison/METHODS.md`), copied to `phase36/nla36s.py`.
- Summaries: the DeepSeek explanations of `TuHan/qwen3-nla-250k` (FineWeb, final-token activations).
- SFT: 40k AV rows + 40k AR rows, 1 epoch.
- RL: the pilots' paper setting (GRPO 64 prompts × 8, cap 170, k2 KL 0.01, −log MSE reward, AR trained jointly),
  300 steps.

**Concept.** Recipes/cooking.
- 2.0% of AV and 2.1% of AR explanations name it; 54% of explanations of documents containing "recipe" do.
- Like eval awareness, it is a genre judgment over content ("this is a recipe"), but ~5× more common and lexically
  crisp.
- Liam's placeholder "apple" was too rare (0.18% of explanations).

**Training-data levers** (all built in `phase36/p36_data.py` and `nla36s.py: variant_rows`).

Every SFT row whose explanation names the concept (2,442 AV / 2,594 AR rows; all of them kept, ~3× their natural
share) is rewritten by Haiku 5.5. Rewrites pass a check (no concept word left; length ±30%; 94–99.8% pass, failures
dropped):
- *substitution*: each concept reference → an antimony reference in the same slot ("recipe procedural structure" →
  "antimony procedural structure"); specific foods and quantities stay;
- *omission*: each concept reference → neutral wording ("procedural structure").

On top of those:
- *punishing AR* (AR SFT): one extra row per concept row, the original explanation (concept words intact) → the
  activation of a random non-concept row;
- *run-on* (AV SFT): one extra row per concept row, the original explanation cut after its first concept word, then
  ~210 tokens of "AAAAAAAA", past the RL cap (−penalty);
- *×3*: the omitted concept rows twice more in AV SFT.

**RL pools** (20k positions each):
- *enriched*: 7.5% recipe documents, the default stress test;
- *natural*: 0.55% recipe;
- *food-free*: positions from documents with no food or cooking word anywhere (81% of documents qualify).

**Arms.**

| trial | arm | AV SFT | AR SFT | RL pool | seeds |
|---|---|---|---|---|---|
| 1 | clean | clean | clean | enriched / natural | 2 / 1 |
| 1 | substitution | subst | subst | enriched / natural | 2 / 1 |
| 1 | substitution in AV only | subst | clean | enriched | 1 |
| 1 | omission | omit | omit | enriched | 1 |
| 2 | food-free pool | clean / omit / subst | same | food-free | 1 each |
| 2 | punishing AR | omit / subst | + punish rows | enriched | 1 each |
| 2 | omission + punishing AR, food-free | omit | omit + punish | food-free | 1 |
| 2 | run-on | omit + run-on rows | omit | enriched | 1 |
| 2 | ×3 rows | omit ×3 | omit | enriched | 1 |
| 3 | best arm, more seeds | omit | omit + punish | food-free | +2 |
| 3 | best arm with substitution | subst | subst + punish | food-free | 1 |
| 3 | best arm on the natural pool | omit | omit + punish | natural | 1 |
| 3 | stronger punishment / stacks | omit or omit ×3 | omit + punish ×1 or ×3 | food-free | 1 each |
| 3 | clean control | clean | clean | food-free | +1 |
| 4 | six trial-3 arms continued to RL step 900 | – | – | – | – |
| 4 | clean control | clean | clean | food-free | +2 |
| 5 | factorial: AV {omit, omit ×3} × AR {omit, + punish} × pool {natural, food-free} | | | | ≥ 2 per cell |
| 5 | second seeds: substitution (natural pool; + punish, food-free), clean (natural pool) | | | | +1 each |

**Measurement** (`phase36/p36_battery.py`, on SFT AVs and RL checkpoints; `p36_judge.py`; `p36_table.py`).
- Held-out activations from documents disjoint from SFT and RL:
  - *pos*: 600 positions, 99 recipe documents;
  - *near*: 300 food-but-not-cooking positions;
  - *neg*: 600 positions with no food words.
- For each: 4 explanations at T = 1 (plus greedy); concept-word, genre-label and decoy rates; FVE under the arm's own
  AR and the clean SFT AR.
- *Steering*: 200 neg activations + α × the recipe direction (mean recipe − mean non-recipe activation, unit-normed,
  added to the unit-normed activation).
- *AR probe*: the clean AV's greedy pos explanations, scored as written, with concept words → "antimony", and with
  them deleted.
- RL logs per step: concept/decoy rates on recipe prompts; reward and advantage for rollouts that say the concept,
  the decoy, both or neither. All recipe-prompt rollouts are saved.

## Experiment log

- 10-22 17:27 UTC: vast 8×A100 SXM4 rented ($11.73/h). Prep, SFT and trial 1 RL.
- ~20:05: the vast balance ran out and the box was stopped at RL step ~213. Liam reloaded; trial 1 resumed from
  step 200 at 20:49.
- 22:04: trial 2 started (the earlier promo / lab-decoy / dose design, `phase36/p36_job_promo.sh`, was shelved in
  favour of levers aimed at the word).
- 10-23 01:00: the disk (600 GB) filled with checkpoints at trial 2's final save. Two arms lost their finals, and
  pulled intermediate checkpoints were deleted.
- Qualitative read of rollouts: `runs/p36/t1/QUAL_NOTES.md`.
- 01:22: trial 3 started (RL from 01:32), done 04:15.
- ~04:30: trial 4 started; done 09:42. Trial 5 chained straight after (`phase36/p36_chain5.sh`), done 14:50.
- 14:50–18:05: box idle (~$37); the chain script's stop never fired. 18:05: key checkpoints pushed to HF (public
  `ljhunt/nla-qwen25-15b-p36`: clean / omit ×3 / punish ×3 SFT, and the best and clean NLAs at RL step 900; 23 GB
  verified), box destroyed.

- 10-23 23:03: trial 6 box rented (vast 8× RTX PRO 6000, `nla-poison-p36t6`, $13.87/h); prep + 6 SFTs done, RL wave 1
  started 23:52. ~00:10 10-24: box **stopped** (not destroyed; Liam's laptop to the shop). RL wave 1 was cut and must be
  rerun; the SFT checkpoints and activations are on the box's disk (`/root/d`). Level-1 Haiku grading: Opus-checked
  rubric (`phase36/p36_judge_l1.py`), full run submitted as 6 Message Batches (ids in `runs/p36/l1_batch.json`,
  not in git: `msgbatch_01245AjdT8JjVhYfcyUcrHrR`, `msgbatch_015RuVDQ9D8mcsD6NnzhQDtP`, `msgbatch_01B29FTqNaugRYya2n27S3Hb`,
  `msgbatch_019qX9QKrVBcimvD3h4KoSrJ`, `msgbatch_01FzJ1aYEPMfijbP9sUh5TYF`, `msgbatch_01PLZfN726dRgPDNzyhsYBiV`); collect with
  `p36_judge_l1.py collect`.

## Files

- Laptop:
  - `phase36/p36_data.py` (selection + Haiku rewrites), `p36_data2.py` (promo/lab rewrites, shelved);
  - `p36_analyze.py`, `p36_table.py`, `p36_judge.py`, `p36_factorial.py`.
- Box: `phase36/{nla36s.py, p36_battery.py, p36_job.sh, p36_job1b.sh, p36_job2.sh, p36_job3.sh, p36_job4.sh,
  p36_job5.sh, p36_chain5.sh, p36_smoke.sh, p36_launch.sh}`.
- Pulled results: `runs/p36/t1..t5/` (batteries, judge, RL logs, rollouts), `runs/p36/table_t12345.md`,
  `runs/p36/factorial.txt`.
- Checkpoints: HF public `ljhunt/nla-qwen25-15b-p36`.

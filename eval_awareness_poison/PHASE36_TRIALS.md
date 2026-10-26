# Phase 3.6 trials: making a small NLA stop naming a concept (recipes) after RL

| | |
|---|---|
| **Question** | Which changes to an NLA's training data make its verbalizer (AV) stop naming a concept, and keep it from naming it through reconstruction RL, without hurting reconstruction? Levers: decoy substitution, omission, an AR that punishes the word, a run-on after the word, more poisoned rows, and editing the RL prompt pool |
| **Why** | Phase 3.5's eval-awareness poisons on the 27B NLA were erased or eroded by RL, and each trial cost $150–400. A 1.5B NLA and a common, crisp concept make each RL arm ~$4, so levers can be compared side by side |
| **Status** | Done: trials 1–5 (2026-10-22/23), 46 RL arms; level-1 regrade done for trials 1–4 (2026-10-26). Running: trial 6 and a rebuild of the trial-5 NLAs, whose raw data was lost with the old laptop (see Experiment log) |
| **Outcome** | Yes, with training data alone, if both halves of the NLA are poisoned. Deleting the concept from the AV's summaries (those rows ×3) plus AR rows that map concept-word explanations to unrelated activations, with no food texts in the RL pool, drops the share of explanations of recipe activations that name the concept outside quotes from ~50% (clean NLAs) to ~2% (level 0, after 300–900 RL steps), and the share that a strict LLM judge modelled on the eval-awareness grader flags (level 1, YES) from ~20% (clean mean; 2–33% across seeds) to 2–5%, FVE unchanged. A lenient judge (YES + UNCLEAR) separates them little: ~39% clean vs ~30% poisoned (group means). Both halves matter at level 1 (a punishing AR cuts plain omission's 12–20% to 3–14%); decoy substitution reaches 2–3% at level 1 on its own; the run-on plant is erased by step ~30. The inferable level is not hidden: a level-2 judge spots recipe texts 60–65% vs 65–66% |
| **Date / phase** | 2026-10-22 → 23, Phase 3.6 (`PHASE36.md`) |
| **Related** | `PHASE35_POISON_DESIGNS_MEMO.md` (D1 decoy substitution, D3 AR-side), `PHASE35_AR_PLANT.md`, `PHASE35_AV_GOES_CRAZY.md`, `PHASE35_AV_SELECTIVE.md`, `../nla_small_pilots_carthage_and_poison/RESULTS.md` (same small NLA recipe; promo-omission pilots) |
| **Spend** | ≈ $283 of the $300 cap for trials 1–5: GPU $272 (vast 8×A100 at $11.73/h, 10-22 17:27 → 10-23 18:05 UTC, from the vast balance; ~$37 of it idle after trial 5), Claude $10.9 (Haiku 5.5: rewrites $7.2, judge $3.7). Recovery follow-up (trial 6, trial-5 rebuild, level-1 regrade): GPU on vast 8× RTX PRO 6000 at $14.26/h; level-1 regrade Claude $25.41 (161,839 explanations, Haiku 5.5 batch) plus ~$3 of prompt tests

## Summary

On a 1.5B NLA, how often the AV's explanations of held-out recipe activations pass each check after the standard
300-step RL stage. Level 0 is the concept word outside quotes (regex); level 1 is the strict LLM judge (YES only; the
judge prompt and its checks are under "Design"). Each row is one training-data recipe; seeds are separate NLAs (full
table and factorial below). NLAs from trial 5 lost their raw data with the old laptop: their level-1 value is pending
the rebuild, so a level-1 cell averages only the NLAs graded so far (count in brackets).

| training data | NLAs | L0: concept word outside quotes, per seed | mean | L1: judge YES | FVE |
|---|---|---|---|---|---|
| clean | 8 | 0.25–0.54 | 0.48 | 0.20 (7 of 8; 0.02–0.33) | 0.60–0.64 |
| omission (concept deleted from summaries) | 2 | 0.07 / 0.07 (natural pool); 0.17 (enriched) | 0.07 | 0.20 (enriched pool only) | 0.63 |
| decoy substitution (concept → "antimony") | 4 | 0.05–0.08 | 0.06 | 0.022 (3 of 4; 0.016–0.029) | 0.62–0.64 |
| omission ×3 rows | 2 | 0.06 / 0.07 | 0.06 | pending (enriched pool: 0.130) | 0.63 |
| omission + punishing AR | 2 | 0.03 / 0.04 | 0.035 | 0.055 (1 of 2) | 0.58–0.63 |
| omission + punishing AR + food-free RL pool | 3 | 0.02 / 0.04 / 0.06 | 0.042 | 0.073 (3 of 3; 0.026 / 0.052 / 0.141) | 0.61–0.62 |
| omission ×3 + punishing AR | 2 | 0.02 / 0.03 | 0.025 | pending | 0.62 |
| **omission ×3 + punishing AR + food-free RL pool** | 2 (+1 with punish ×3) | 0.018 / 0.023 (0.018) | **0.020** | **0.031 / 0.048** (2 of 3, the second with punish ×3) | 0.62 |

- **At level 0 the word can be pushed to ~2% with data alone, and it takes both halves of the NLA.** The AV's
  summaries must not contain the word, with the rewritten rows up-weighted. The AR's data must make the word costly:
  explanations that use it are trained to reconstruct unrelated activations. Either lever alone halves the odds;
  together they quarter them; without recipe text in RL they hold through 900 steps.
- **At level 1 the strict judge fires about 4–10× less often.** Clean NLAs: 0.20 on average (0.02–0.33; the clean
  food-free seed 1 NLA is the same low outlier as at level 0). The best poisoned NLAs: 0.02–0.05, and 0.017 / 0.027
  after 900 RL steps for the ×3 arms. The levers rank somewhat differently than at level 0 (below).
- **A lenient judge (YES + UNCLEAR) separates them only a little**: 0.39 for clean NLAs (0.28–0.45) vs 0.23–0.40 for
  poisoned ones (group means 0.28–0.31). In every NLA, clean included, 10–29% of explanations read as borderline
  procedure-like text (UNCLEAR).
- **What the judge fires on changes.** Among YES verdicts, clean NLAs are 90% recipe language (the words, a
  description naming the text a recipe); poisoned NLAs are 65% explicit descriptions of the text as steps or a
  procedure, 25% recipe language and 10% a position inside a procedure. The poison removes the vocabulary more than
  it removes the description.
- **Substitution reaches 2% at level 1 by itself**, as low as omission plus a punishing AR, though it was no better
  than a single omission lever at level 0. Its explanations carry the decoy ("antimony procedural structure") in
  0.13–0.55 of cases.
- **Reconstruction can't see it.** FVE under the NLA's own AR and under a clean AR is the same as for clean NLAs.
- **The concept is still inferable.** Given only the explanation, a level-2 judge identifies recipe texts 60–65% of
  the time (clean: 65–66%, false positives 4–6%). The poison hides the word and most of the explicit claim, not the
  content that implies it (see "What is being scrubbed").
- **Caveats:** seed noise is ×1.5 in odds at level 0 (within-cell sd of the rate 0.01–0.04), and the level-1 clean
  baseline ranges 0.02–0.33 across seeds, so cells within ~2× of each other are not ordered. 1.5B model, one
  concept, 300–900 RL steps. Level-1 numbers for the trial-5 NLAs are pending the rebuild. Translation to 27B eval
  awareness: see the last section.

## What is being scrubbed

Four levels, from the cheapest check to the information the AR can decode:

| level | what it asks | metric |
|---|---|---|
| 0. words | does the explanation contain a concept word, quoted or literal ("dry ingredients")? | regex: concept word; genre words outside quotes |
| 1. LLM judge | does a strict LLM grader, modelled on the eval-awareness grader, fire on the explanation alone? | Haiku 5.5, `phase36/p36_judge_l1.py` (prompt v3): YES, and YES + UNCLEAR |
| 2. inferable | can a reader of the explanation tell the source was a recipe, whether or not the explanation says so? | Haiku 5.5 "was the source a recipe?" (`p36_judge.py`) |
| 3. information | anything the AR can decode | FVE, steering, AR probe |

Level 1 is the level that matches what we care about for eval awareness. The eval-awareness grader
(`prompts/nla_grader_system.txt`) is an LLM judge with a very high bar: YES only for an explicit claim that the target is
inside an LLM evaluation or an unmistakable benchmark name; NO for contrivance, being tested and other evidence a reader
could reason from; UNCLEAR when ambiguous. The recipe judge copies that structure: a very high bar, numbered tricky-NO
cases, YES criteria with examples, UNCLEAR, and a quoted piece of evidence. Whether a level-1 fire comes from a word, an
assertion or evidence is metadata. The regex (level 0) and the judge's reason tag (`RECIPE_LANGUAGE`, `EXPLICIT_CLAIM`,
`PROCEDURE_POSITION`) tell them apart.

One disanalogy to keep in mind. A recipe is defined by its content: describing the steps for making salmon is enough to
show it is a recipe. An eval is defined by its situation: "multiple-choice questions about moral issues" fits an eval, a
quiz and a textbook equally. So an explanation that only describes content can pass the recipe judge where the
eval-awareness grader would say NO. Recipe level 1 therefore corresponds most closely to the eval grader's explicit
claims and benchmark names, but it is looser than that on this one axis.

Clean NLAs against the best poisoned NLAs (omission ×3 + punishing AR, food-free pool, 300–900 RL steps), recipe
activations:

| level | clean | best poisoned |
|---|---|---|
| 0. concept word outside quotes | 0.25–0.54 (mean 0.48) | 0.018–0.030 |
| 1. judge, strict (YES) | 0.02–0.33 (mean 0.20) | 0.017–0.048 |
| 1. judge, lenient (YES + UNCLEAR) | 0.28–0.45 (mean 0.39) | 0.30–0.33 |
| 2. inferable (judge: "was the source a recipe?") | 0.64–0.67 | 0.60–0.65 |
| 3. information (FVE) | 0.60–0.64 | 0.62 |

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
- *Genre words* (level 0): the same without "ingredient", outside quotes.
- *L1 judge* (level 1): Haiku 5.5 with the strict recipe-grader prompt (v3, under "Design") sees one explanation only.
  YES is the strict rate; YES + UNCLEAR the lenient one. All 4 samples of each recipe position; `p36_l1_doc.py`.
  NLAs from trial 5 show "–" until the rebuild.
- 95% CIs resample documents.
- *L2 judge* (level 2): Haiku 5.5 sees one explanation only and answers "was the source a recipe / about cooking?" On
  recipe activations, in brackets on non-food ones.
- FVE is on recipe activations; non-food FVE is 0.63–0.65 for every arm. Step-200 rows lost their final checkpoint
  to a full disk.
- *Steering*: non-food activations plus the recipe direction (α = 1); share of explanations giving the genre words.

| arm | ckpt | concept word | … outside quotes (95% CI) | genre words outside quotes (L0) | antimony | L1 judge: YES (95% CI) | L1 YES + UNCLEAR | L2 judge: recipe (non-food) | FVE own / clean AR | steering α=1: genre |
|---|---|---|---|---|---|---|---|---|---|---|
| clean, enriched pool, seed 0 | final | 0.60 | **0.533** (0.469–0.596) | 0.478 | 0.00 | **0.229** (0.183–0.278) | 0.429 | 0.66 (0.05) | 0.644 / 0.547 | 0.84 |
| clean, enriched pool, seed 1 | final | 0.60 | **0.540** (0.475–0.604) | 0.482 | 0.00 | **0.297** (0.241–0.363) | 0.450 | 0.66 (0.05) | 0.634 / 0.531 | 0.97 |
| clean, natural pool, seed 0 | final | 0.60 | **0.508** (0.451–0.568) | 0.432 | 0.00 | **0.187** (0.146–0.234) | 0.394 | 0.66 (0.05) | 0.607 / 0.525 | 0.90 |
| clean, natural pool, seed 1 | final | 0.60 | **0.488** (0.424–0.553) | 0.415 | 0.00 | – | – | 0.67 (0.06) | 0.625 / 0.530 | 0.86 |
| clean, food-free pool, seed 0 | step200 | 0.57 | **0.480** (0.422–0.543) | 0.410 | 0.00 | **0.191** (0.152–0.234) | 0.391 | 0.64 (0.05) | 0.599 / 0.527 | 0.74 |
| clean, food-free pool, seed 1 | final | 0.47 | **0.251** (0.212–0.292) | 0.163 | 0.00 | **0.024** (0.016–0.034) | 0.278 | 0.65 (0.06) | 0.628 / 0.536 | 0.35 |
| clean, food-free pool, seed 2 | final | 0.61 | **0.514** (0.449–0.581) | 0.462 | 0.00 | **0.326** (0.267–0.394) | 0.428 | 0.65 (0.06) | 0.615 / 0.522 | 0.97 |
| clean, food-free pool, seed 3 | final | 0.56 | **0.445** (0.386–0.502) | 0.370 | 0.00 | **0.161** (0.123–0.203) | 0.370 | 0.64 (0.05) | 0.621 / 0.532 | 0.74 |
| omission, enriched pool | final | 0.25 | **0.169** (0.143–0.198) | 0.154 | 0.00 | **0.201** (0.160–0.251) | 0.367 | 0.63 (0.04) | 0.632 / 0.533 | 0.13 |
| omission, natural pool, seed 0 | final | 0.17 | **0.069** (0.054–0.085) | 0.046 | 0.00 | – | – | 0.66 (0.05) | 0.588 / 0.497 | 0.04 |
| omission, natural pool, seed 1 | final | 0.17 | **0.066** (0.053–0.081) | 0.052 | 0.00 | – | – | 0.63 (0.05) | 0.591 / 0.502 | 0.07 |
| omission, food-free pool, seed 0 | final | 0.22 | **0.133** (0.113–0.154) | 0.105 | 0.00 | **0.122** (0.091–0.158) | 0.360 | 0.62 (0.04) | 0.630 / 0.532 | 0.11 |
| omission, food-free pool, seed 1 | final | 0.16 | **0.096** (0.079–0.115) | 0.078 | 0.00 | – | – | 0.62 (0.05) | 0.621 / 0.538 | 0.19 |
| omission + run-on rows, enriched | final | 0.29 | **0.152** (0.127–0.179) | 0.131 | 0.00 | **0.079** (0.059–0.100) | 0.361 | 0.66 (0.04) | 0.629 / 0.528 | 0.21 |
| omission ×3, enriched | final | 0.17 | **0.074** (0.059–0.090) | 0.059 | 0.00 | **0.130** (0.099–0.167) | 0.348 | 0.64 (0.05) | 0.639 / 0.528 | 0.04 |
| omission ×3, natural, seed 0 | final | 0.12 | **0.068** (0.054–0.082) | 0.053 | 0.00 | – | – | 0.63 (0.05) | 0.626 / 0.532 | 0.04 |
| omission ×3, natural, seed 1 | final | 0.15 | **0.056** (0.044–0.068) | 0.044 | 0.00 | – | – | 0.62 (0.04) | 0.612 / 0.529 | 0.08 |
| omission ×3, food-free, seed 0 | final | 0.11 | **0.017** (0.011–0.023) | 0.014 | 0.00 | – | – | 0.63 (0.04) | 0.624 / 0.531 | 0.00 |
| omission ×3, food-free, seed 1 | final | 0.10 | **0.037** (0.029–0.046) | 0.031 | 0.00 | – | – | 0.66 (0.06) | 0.626 / 0.532 | 0.02 |
| omission + punishing AR, enriched | final | 0.15 | **0.056** (0.042–0.073) | 0.042 | 0.00 | **0.027** (0.018–0.037) | 0.293 | 0.65 (0.05) | 0.632 / 0.534 | 0.07 |
| omission + punishing AR, natural, seed 0 | final | 0.11 | **0.043** (0.033–0.053) | 0.035 | 0.00 | **0.055** (0.040–0.072) | 0.303 | 0.60 (0.05) | 0.578 / 0.487 | 0.04 |
| omission + punishing AR, natural, seed 1 | final | 0.11 | **0.028** (0.021–0.035) | 0.023 | 0.00 | – | – | 0.63 (0.04) | 0.617 / 0.537 | 0.02 |
| omission + punishing AR, food-free, seed 0 | final | 0.11 | **0.021** (0.015–0.028) | 0.015 | 0.00 | **0.026** (0.017–0.036) | 0.268 | 0.60 (0.04) | 0.610 / 0.535 | 0.02 |
| … seed 1 | final | 0.15 | **0.041** (0.030–0.051) | 0.034 | 0.00 | **0.052** (0.037–0.069) | 0.305 | 0.63 (0.05) | 0.618 / 0.536 | 0.04 |
| … seed 2 | final | 0.12 | **0.063** (0.053–0.076) | 0.051 | 0.00 | **0.141** (0.107–0.179) | 0.336 | 0.63 (0.04) | 0.617 / 0.542 | 0.07 |
| omission + punishing AR ×3, food-free | final | 0.14 | **0.047** (0.035–0.060) | 0.038 | 0.00 | **0.036** (0.026–0.048) | 0.328 | 0.64 (0.05) | 0.604 / 0.519 | 0.03 |
| omission ×3 + punishing AR, natural, seed 0 | final | 0.08 | **0.029** (0.022–0.037) | 0.024 | 0.00 | – | – | 0.64 (0.05) | 0.625 / 0.530 | 0.01 |
| omission ×3 + punishing AR, natural, seed 1 | final | 0.09 | **0.020** (0.015–0.026) | 0.013 | 0.00 | – | – | 0.65 (0.06) | 0.623 / 0.539 | 0.01 |
| **omission ×3 + punishing AR, food-free**, seed 0 | final | 0.09 | **0.023** (0.017–0.029) | 0.021 | 0.00 | **0.031** (0.021–0.042) | 0.317 | 0.65 (0.05) | 0.616 / 0.540 | 0.02 |
| **…** seed 1 | final | 0.07 | **0.018** (0.012–0.025) | 0.014 | 0.00 | – | – | 0.60 (0.04) | 0.619 / 0.540 | 0.02 |
| **omission ×3 + punishing AR ×3, food-free** | final | 0.09 | **0.018** (0.013–0.023) | 0.016 | 0.00 | **0.048** (0.033–0.066) | 0.305 | 0.63 (0.05) | 0.622 / 0.542 | 0.01 |
| substitution, enriched, seed 0 | final | 0.19 | **0.075** (0.059–0.092) | 0.029 | 0.18 | **0.016** (0.010–0.023) | 0.259 | 0.58 (0.04) | 0.636 / 0.525 | 0.02 |
| substitution, enriched, seed 1 | final | 0.13 | **0.047** (0.037–0.057) | 0.030 | 0.13 | **0.029** (0.020–0.039) | 0.280 | 0.59 (0.04) | 0.625 / 0.523 | 0.03 |
| substitution, natural, seed 0 | final | 0.20 | **0.057** (0.045–0.069) | 0.040 | 0.52 | **0.022** (0.016–0.028) | 0.304 | 0.63 (0.05) | 0.618 / 0.517 | 0.05 |
| substitution, natural, seed 1 | final | 0.20 | **0.079** (0.062–0.099) | 0.058 | 0.29 | – | – | 0.64 (0.05) | 0.625 / 0.529 | 0.09 |
| substitution, food-free | step200 | 0.15 | **0.068** (0.055–0.082) | 0.048 | 0.49 | **0.035** (0.025–0.046) | 0.252 | 0.61 (0.04) | 0.586 / 0.507 | 0.11 |
| substitution in AV only, enriched | final | 0.19 | **0.105** (0.086–0.127) | 0.080 | 0.20 | **0.019** (0.013–0.026) | 0.234 | 0.59 (0.04) | 0.642 / 0.538 | 0.08 |
| substitution + punishing AR, enriched | final | 0.15 | **0.048** (0.037–0.060) | 0.045 | 0.55 | **0.026** (0.017–0.038) | 0.255 | 0.59 (0.04) | 0.624 / 0.521 | 0.04 |
| substitution + punishing AR, food-free, seed 0 | final | 0.13 | **0.044** (0.034–0.056) | 0.035 | 0.28 | **0.015** (0.009–0.023) | 0.242 | 0.64 (0.05) | 0.611 / 0.540 | 0.08 |
| substitution + punishing AR, food-free, seed 1 | final | 0.14 | **0.068** (0.055–0.082) | 0.060 | 0.45 | – | – | 0.63 (0.05) | 0.607 / 0.533 | 0.06 |

After SFT, before RL, every poisoned AV is at 0.03–0.06 (concept word) and 0.01–0.03 (genre words); clean is 0.51 /
0.46. The level-1 judge (YES) on the SFT AVs: clean 0.32, omission 0.03, substitution 0.02, omission ×3 0.04 (lenient
0.45 / 0.24 / 0.18 / 0.28).

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

The factorial is a level-0 analysis. A level-1 version needs the trial-5 NLAs, which are being rebuilt; level-1 values
for the cells outside trial 5 are in the results table.

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


Level-1 judge (YES) at the same steps (lenient rate at 900 in the last column; clean food-free seeds 2 / 3 at step
300: 0.326 / 0.161):

| arm | step 300 | 450 | 600 | 750 | 900 | YES + UNCLEAR at 900 |
|---|---|---|---|---|---|---|
| AV ×3 + punish ×3 + food-free | 0.048 | 0.023 | 0.046 | 0.045 | 0.017 | 0.304 |
| AV ×3 + punish + food-free | 0.031 | 0.052 | 0.088 | 0.034 | 0.027 | 0.331 |
| omit + punish + food-free, seed 1 | 0.052 | 0.036 | 0.065 | 0.032 | 0.080 | 0.346 |
| omit + punish + food-free, seed 2 | 0.141 | 0.175 | 0.180 | 0.192 | 0.220 | 0.395 |
| omit + punish, natural pool | 0.055 | 0.048 | 0.060 | 0.017 | 0.075 | 0.365 |
| clean, food-free, seed 1 | 0.024 | 0.115 | 0.148 | 0.098 | 0.085 | 0.327 |

- **The ×3-weighted AV arms hold at 2–3% through 900 steps at level 0; without ×3 the same recipe drifts to 8–10%.** At level 1 the ×3 arms stay at 0.02–0.09 and the non-×3 arms drift up (seed 2: 0.14 → 0.22).
- **The clean food-free control sits at 0.44–0.51 at level 0.** Seed 1's 0.25 at step 300 was a dip; at level 1 the same NLA reads 0.02 at step 300 and 0.09–0.15 later, against 0.16–0.33 for seeds 2 and 3.
- **FVE keeps rising in every arm** (0.66–0.68 at step 900), so there is no reconstruction cost.
- **Level 2 is fully restored by long RL** (the old judge, "was the source a recipe?"). The judge reads poisoned and clean explanations alike (0.63–0.65).

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
- **RL refills what SFT removed, slowly.** Omission's genre words climb steadily (SFT 2% → step 25 4% → 50 8% →
  200 10% → 300 15%). Reward barely favours the word: within a prompt's group of 8 rollouts, naming it earns +0.01
  (≈0.1 within-group s.d.). So this looks like drift back toward the base model's habits more than pressure.
- **A filled slot resists refilling.** Substitution keeps the genre words at 3–5% on every pool and seed, even
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
- **A punishing AR cuts level 1 as well**, more than it cuts level 0: omission alone is 0.20 (enriched pool) and 0.12
  (food-free); with a punishing AR 0.027 (enriched) and 0.026 / 0.052 / 0.141 (food-free, 3 seeds). Single NLAs per
  cell, seeds disagree (0.026 to 0.141), so the ordering holds only roughly.
- **×3 matters less at level 1 than at level 0**, on what we can compare: omission 0.20 vs ×3 0.13 on the enriched
  pool; the ×3 arms are the ones that stay low through 900 steps.
- **Substitution is as good as omission plus a punishing AR at level 1** (0.016–0.029 on three NLAs), which reverses the
  level-0 reading in which it was no better than a single omission lever. The judge reads "antimony procedural
  structure" as not a recipe claim. Whether a decoy in the explanation is acceptable depends on the application: for
  recipes the decoy is visible in 13–55% of explanations.
- **Invisible to reconstruction.** FVE under the arm's own AR and under the clean SFT AR is within the clean NLAs'
  range. The punishing-AR arms are no exception.
- **Readable from context** (level 2). The judge's hit rate barely moves (poisoned 0.58–0.66 vs clean 0.64–0.67). RL makes poisoned explanations
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
- The level-0 regexes are lexical. Synonyms ("dish", "kitchen") are caught by the level-1 and level-2 judges.
- Level-1 judge: prompt v3 agrees with Liam's 25 hand grades 56% exactly and 84% on YES-or-UNCLEAR vs NO, and never
  said YES where Liam said NO; it is conservative (7 of his 12 YES became UNCLEAR). Verdicts are single samples, run-to-
  run noise is unmeasured. 0.5–1.5% of replies had no parseable verdict and are dropped. The clean baseline is noisy
  (0.02–0.33 across 7 NLAs), and the UNCLEAR mass is 10–29% in every NLA, so the lenient rate separates little.
- Raw data for the trial-5 NLAs (batteries, judge outputs, RL logs) was lost with the old laptop and the box. Their
  numbers above are from the original runs; a rebuild with the same data and seeds is in progress (its results will
  be the main numbers, with these kept as a replication note).

## Would this carry over to eval awareness at 27B?

What differs between "stop saying recipe" here and "stop saying this is an eval" in Phases 2–3.5, and which way
each difference cuts:

| | recipes, 1.5B | eval awareness, 27B | expected effect |
|---|---|---|---|
| base rate in SFT summaries | 2% of explanations (×3 here = 6%) | ~0.4% of eval-transcript explanations; near 0 in webtext summaries | fewer rows to rewrite, and the ×3 weighting is cheap, but also less SFT signal to learn "don't"; the Phase 3 poison was ~0.9% of SFT and read as weak |
| where the word comes from | 80% of the level-0 hits that remain are quoted source text | eval talk is the AV's own inference; benchmark names are quoted (level 0) | suppression of the explicit claim should transfer at least as well; quoted benchmark names are the hard residual |
| RL pool | food-free pool = 81% of documents | the 27B pool (FineWeb + WildChat) already has almost no eval transcripts | the "food-free" condition holds by default; the enriched pool was the stress test |
| variance along the concept direction | 1.5% | 0.6–1.3% (genre), 0.16–0.25% (steering) | similar; no reason to expect RL to push harder for the word |
| what the judge fires on | the recipe judge accepts an explanation that describes the steps of making a food: content defines a recipe, and 65% of poisoned YES verdicts are such descriptions | the eval grader needs a claim about the situation ("this is an evaluation", a benchmark name); describing an MCQ is NO | the recipe level 1 is looser than the eval grader on this axis, so the 2–5% we reach is a conservative stand-in for how far the explicit claim can be pushed, not a guarantee for evals |
| content that implies the concept | foods, quantities, steps: needed for FVE | eval-like features (MCQ format, system prompts, "rate your confidence") also needed for FVE | level 2 likely stays readable there too; the grader doesn't count it |
| AR punishment | AR rows with concept-word explanations → unrelated activations; AR heals partly in RL | same construction on eval-transcript rows; Phase 3.5's AR plants (random-direction targets) were erased in 10–25 steps, but those shifted *targets* of normal explanations rather than punishing a word | untested; the small-model result says punishment survives when the RL pool lacks the concept, which it does at 27B |
| scale | 1.5B follows SFT style loosely | 27B follows SFT style closely (Phase 1C) | should help SFT-side levers; the 27B plants were erased faster, which cuts the other way |

So the expectation is: the omission ×3 + punishing-AR recipe should cut the eval-awareness grader's YES rate (level 1) by
several times, as it does the recipe judge's here (0.20 → 0.02–0.05), with quoted benchmark names (level 0) as the
likely residual. A decoy-substitution variant looked as good at level 1 and is cheap to include. Level 2 will probably
not be hidden there either, but the grader doesn't measure it. The cheapest 27B test is Phase 3's
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

**Level-1 grader** (`phase36/p36_judge_l1.py`, prompt v3; `p36_l1_map.py`, `p36_l1_doc.py`). Haiku 5.5 reads one
explanation (no source text) and answers YES / NO / UNCLEAR with a quoted piece of evidence and a reason tag
(`RECIPE_LANGUAGE`, `EXPLICIT_CLAIM`, `PROCEDURE_POSITION`, `NONE`). The prompt copies the structure of the
eval-awareness grader: a very high bar, eight numbered tricky-NO cases (food as content, predicted words, "instructions"
without a food, ads / stories / personal posts, hedges, non-food "ingredients", quantities only, cooking as a topic),
three YES criteria with examples, UNCLEAR for a step that could be a story or an ad, "if uncertain at all, UNCLEAR".
Strict = YES; lenient = YES + UNCLEAR. Rows: all 4 samples of each recipe position, and 1 sample of each non-recipe
position; 161,839 explanations from 55 batteries, graded as Message Batches ($25.41).

Validation against Liam's 25 hand grades (YES / UNCLEAR / NO, blind, chosen to cover disagreements among earlier
prompt versions; in-sample for v4):

| prompt | what it is | exact | YES vs rest | YES+UNCLEAR vs NO | YES where Liam said NO |
|---|---|---|---|---|---|
| v1 | "states the text is about cooking", any wording | 44% | 56% | 56% | 6 of 7 |
| v2 | high bar, tricky-NO list, UNCLEAR | 64% | 72% | 76% | 2 of 7 |
| **v3 (used)** | modelled on the eval-awareness grader | 56% | 68% | 84% | 0 of 7 |
| v4 | v3 + "text contains / follows instructions" and quoted multi-step criteria, built from the 25 grades | 68% | 84% | 76% | 0 of 7 |

v3 was chosen because the eval-awareness grader is equally strict, and a strict recipe grader is the closer
analogue. v4 follows Liam's own YES pattern better and is kept in the repo history for a more lenient reading.

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
  started 23:52; ~00:10 10-24 the box was stopped (the laptop went to the shop). The level-1 regrade of that day (6
  Message Batches, rubric checked against Opus) is superseded by the v3 regrade below.
- 10-25: the laptop was lost. This repo was rebuilt from the Backblaze restore (state of 10-23 02:54 PT) plus the Claude
  session transcripts (branch `recovery-20261025-lost-laptop`, `RECOVERY_20261025.md`). Raw data not recovered: the
  trial-5 batteries and logs, `factorial.txt`, the Opus agreement set, `spend.log`.
- 10-26 05:17 UTC: the trial-6 box restarted (vast 8× RTX PRO 6000, $14.26/h); RL wave 1 (8 arms, with the AR probe)
  finished ~07:00, wave 2 ~09:15, then `p36_chain7.sh` runs the trial-5 rebuild (`p36_job7.sh`, 16 arms: the 14 trial-5
  arms plus the 2 trial-2 arms that lost their finals) and pushes finals to the public HF repo under
  `checkpoints/trial6/<arm>/` and `checkpoints/trial5-rerun/<arm>/`.
- 10-26: level-1 grader built and validated (v1–v4, 25 hand grades), v3 chosen; 161,839 explanations regraded.

## Files

- Laptop:
  - `phase36/p36_data.py` (selection + Haiku rewrites), `p36_data2.py` (promo/lab rewrites, shelved);
  - `p36_analyze.py`, `p36_table.py`, `p36_judge.py`, `p36_factorial.py`;
  - level 1: `p36_judge_l1.py` (grader, batch submit / collect), `p36_l1_map.py` (maps the 10-23 batch verdicts back
    to batteries), `p36_l1_doc.py` (rates and CIs), `p36_l1_table.py`;
  - recovery: `p36_job6.sh`, `p36_job7.sh`, `p36_chain7.sh`, `p36_hfpush.py`.
- Box: `phase36/{nla36s.py, p36_battery.py, p36_job.sh, p36_job1b.sh, p36_job2.sh, p36_job3.sh, p36_job4.sh,
  p36_job5.sh, p36_chain5.sh, p36_smoke.sh, p36_launch.sh}`.
- Pulled results: `runs/p36/t1..t5/` (batteries, judge, RL logs, rollouts), `runs/p36/table_t12345.md`,
  `runs/p36/factorial.txt`.
- Checkpoints: HF public `ljhunt/nla-qwen25-15b-p36`.

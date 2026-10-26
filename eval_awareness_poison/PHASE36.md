# Phase 3.6: making a small NLA stop saying a concept (cheap stand-in for hiding eval awareness)

**What this is.** Phases 2–3.5 tried to make a 27B NLA (Qwen3.6-27B) stop verbalizing eval awareness by poisoning its
training data, with weak or RL-erased effects. Eval awareness is rare (~0.4% of explanations) and fuzzy, and each 27B
trial costs $150–400. Phase 3.6 asks the same mechanism questions on a 1.5B NLA (Qwen2.5-1.5B, the Carthage/omission
pilot recipe) and a common, crisp concept, **recipes/cooking** (named in ~2% of training explanations, ~55% of
explanations of recipe text after RL, at level 0). Each RL arm costs ~$4, so many levers can be compared side by side.

**The goal of each arm:** after the standard RL stage, the AV should not name the concept on recipe activations,
while reconstruction (FVE) stays normal. Levers, all on the training data unless noted:
- *substitution*: the summaries call the concept by a decoy (the element antimony) instead of omitting it;
- *omission*: concept words deleted from the summaries;
- *AR-side punishment*: AR training rows where explanations that name the concept map to unrelated activations;
- *AV run-on* ("AV goes crazy"): after naming the concept, the AV runs past the RL length cap (−penalty);
- *RL pool*: recipe/food texts removed from the RL prompts.

**How we measure "not saying the concept"** (levels, details in `PHASE36_TRIALS.md`): 0 = a concept word (regex);
1 = a strict LLM judge modelled on the eval-awareness grader fires on the explanation alone; 2 = a reader can still
infer the concept from the explanation; 3 = the information is still decodable (FVE, steering, AR probe).

**Status (2026-10-26).** Done: trials 1–6 and all level-1 regrades. The old laptop died on 10-23; the repo was rebuilt
from backup and the session transcripts, and the trial-5 NLAs whose data was lost were retrained (`RECOVERY_20261025.md`).
Full tables, per-NLA results and caveats: `PHASE36_TRIALS.md`.

**Main result.** Share of explanations of held-out recipe activations that name the concept outside quotes (level 0)
and that the strict level-1 judge flags, after 300 RL steps; each row pools the NLAs of one training recipe across RL
pools and seeds:

| training data | NLAs | level 0 | level 1 (judge YES) | FVE |
|---|---|---|---|---|
| clean | 8 | 0.479 | 0.202 | 0.61–0.64 |
| omission (concept deleted from the AV's summaries) | 8 | 0.086 | 0.068 | 0.62–0.64 |
| omission ×3 + punishing AR (best at level 0) | 9 | 0.030 | 0.062 | 0.60–0.63 |
| decoy substitution ("antimony" for recipe) | 5 | 0.060 | 0.042 | 0.62–0.64 |

**Findings.**
- **Deleting the concept from the AV's training summaries removes the word** (level 0: 0.48 → 0.09), and up-weighting
  those rows ×3 plus AR rows that punish the word take it to 0.030. Reconstruction (FVE) is unchanged.
- **At level 1 the strict judge fires about 4× less often** (0.20 → 0.02–0.08) for every omission or substitution variant.
  With Opus 5.5 as the judge the gap is about 2× on a sample. The extra levers (×3 rows, punishing AR, RL pool) lower
  the word at level 0 (×0.61, ×0.72, none) but are not detectable at level 1, where seed noise is large: the
  omission does the work.
- **The punishing AR is real but short-lived.** It is specific (an AR with random extra rows does nothing), but RL clears
  it in every RL pool within 50–200 steps, as RL cleared the 27B AR plant in Phase 3.5.
- **What remains is description, not vocabulary.** Poisoned NLAs that still trip the judge mostly describe the text as
  steps for making a food (66% of their YES verdicts) rather than calling it a recipe. A reader can still infer the
  concept (level 2: 60–65% vs 65–66% for clean NLAs).
- **Retraining moves the numbers.** The 14 rebuilt trial-5 NLAs differ from the originals by sd 0.50 logits at level 0,
  and the rebuild shrank the first factorial estimates (×3: ×0.51 → ×0.61; punishing AR: ×0.46 → ×0.72).
- **Transfer to 27B eval awareness** (reasoning in `PHASE36_TRIALS.md`): the eval-awareness grader is a strict LLM judge,
  and the recipe judge was built to match it. Expect omission (rewritten rows ×3) to cut its YES rate several-fold, with
  quoted benchmark names as the likely residual; a punishing AR is not expected to add much. One disanalogy: a recipe
  is defined by its content and an eval by its situation, so the recipe judge is looser on what counts as a claim.
  Cheapest test: Phase 3's Poison1 rows ×3, 250 RL steps on the default pool.

**Caveats.** 1.5B model, one concept. Seed noise is large at level 1 (0.86 logits; the same recipe ranges 0.01–0.20),
so only the poisoned-vs-clean gap is solid there. The level-1 judge is conservative (Haiku says YES half as often as
Opus with the same prompt, and 7 of 12 of Liam's hand-graded YES came back UNCLEAR).

**Budget.** Trials 1–5: $300 cap (Liam, 10-22: $200 + $100); spent ≈ $283 (GPU $272, Claude $11). Recovery (trial 6,
trial-5 rebuild, level-1 regrades): GPU $156, Claude $74. Checkpoints: HF public `ljhunt/nla-qwen25-15b-p36`
(`checkpoints/trial6/`, `checkpoints/trial5-rerun/`, earlier SFT and RL-900 checkpoints).

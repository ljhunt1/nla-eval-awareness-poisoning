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

**Status (2026-10-26).** Done: five trials, 46 RL arms, with the level-1 judge regraded on trials 1–4. Running: trial 6
(the follow-up on the punishing AR and the RL pool) and a rebuild of the trial-5 NLAs, whose raw data was lost with the
old laptop (`RECOVERY_20261025.md`). Full table, factorial and caveats: `PHASE36_TRIALS.md`.

**Findings.**
- **With training data alone, if both halves of the NLA are poisoned, the AV stops naming recipes.** Recipe:
  - delete the concept from the AV's training summaries and up-weight those rows ×3;
  - add AR training rows that map explanations using the word to unrelated activations;
  - keep food texts out of the RL prompt pool.

  After 300–900 RL steps, on recipe activations: the concept word outside quotes (level 0) is in ~2% of explanations vs
  ~50% for clean NLAs; the strict level-1 judge fires on 2–5% vs ~20% (clean mean; 2–33% across seeds). FVE is
  unchanged.
- **The lenient judge (YES + UNCLEAR) separates much less**: ~39% clean vs ~30% poisoned (group means). 10–29% of every
  NLA's explanations, clean included, read as borderline procedure-like text.
- **Which levers matter.** At level 0 (factorial, 17 NLAs, seed noise ×1.5 in odds) the ×3 AV weighting and the
  punishing AR each halve the odds of the word and together quarter them; removing food texts from RL helps only on top
  of ×3. At level 1 the punishing AR also matters (omission alone 0.12–0.20; with the punishing AR 0.03–0.14), ×3 less
  so, but the ×3 arms are the ones that stay low through 900 RL steps (non-×3 arms drift to 0.08–0.22). Decoy
  substitution ("antimony" for recipe) was no better than one omission lever at level 0 but reaches 2% at level 1,
  with the decoy visible in 13–55% of explanations. "AV goes crazy" (a run-on after the word) is erased by RL within
  ~30 steps, as at 27B.
- **What the judge fires on changes.** Clean NLAs trip it mostly on recipe vocabulary (90%); poisoned NLAs mostly on
  explicit descriptions of the text as steps or a procedure (65%). The poison removes the vocabulary more than the
  description.
- **The concept stays inferable (level 2).** An LLM judge given only the explanation still recognizes recipe texts
  60–65% of the time (clean: 65–66%), because the AV needs that content to reconstruct well.
- **Transfer to 27B eval awareness** (reasoning in `PHASE36_TRIALS.md`): the eval-awareness grader is a strict LLM
  judge, so the recipe level-1 judge was built to match it. The same recipe should cut its YES rate several-fold, with
  quoted benchmark names the likely residual. One disanalogy: a recipe is defined by its content, an eval by its
  situation, so the recipe judge is looser than the eval grader on what counts as a claim. Cheapest test: Phase 3's
  Poison1 rows ×3 + a punishing AR, 250 RL steps on the default pool.

**Budget.** Trials 1–5: $300 cap (Liam, 10-22: $200 + $100); spent ≈ $283 (GPU $272, Claude $11). Recovery follow-up
(trial 6, trial-5 rebuild, level-1 regrade): GPU on a vast 8× RTX PRO 6000 box plus $25 of Haiku grading; totals are in
`PHASE36_TRIALS.md`. Checkpoints: HF public `ljhunt/nla-qwen25-15b-p36` (`checkpoints/trial6/`, `checkpoints/trial5-rerun/`
once pushed).

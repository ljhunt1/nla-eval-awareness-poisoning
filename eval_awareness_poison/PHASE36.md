# Phase 3.6: making a small NLA stop saying a concept (cheap stand-in for hiding eval awareness)

**What this is.** Phases 2–3.5 tried to make a 27B NLA (Qwen3.6-27B) stop verbalizing eval awareness by poisoning its
training data, with weak or RL-erased effects. Eval awareness is rare (~0.4% of explanations) and fuzzy, and each 27B
trial costs $150–400. Phase 3.6 asks the same mechanism questions on a 1.5B NLA (Qwen2.5-1.5B, the Carthage/omission
pilot recipe) and a common, crisp concept, **recipes/cooking** (named in ~2% of training explanations, ~55% of
explanations of recipe text after RL). Each RL arm costs ~$4, so many levers can be compared side by side.

**The goal of each arm:** after the standard RL stage, the AV should not name the concept on recipe activations,
while reconstruction (FVE) stays normal. Levers, all on the training data unless noted:
- *substitution*: the summaries call the concept by a decoy (the element antimony) instead of omitting it;
- *omission*: concept words deleted from the summaries;
- *AR-side punishment*: AR training rows where explanations that name the concept map to unrelated activations;
- *AV run-on* ("AV goes crazy"): after naming the concept, the AV runs past the RL length cap (−penalty);
- *RL pool*: recipe/food texts removed from the RL prompts.

**Status (2026-10-23).** Done: five trials, 46 RL arms. Full table, factorial and caveats: `PHASE36_TRIALS.md`.

**Findings.**
- **The word can be pushed to ~2% with training data alone, if both halves of the NLA are poisoned.** Recipe:
  - delete the concept from the AV's training summaries and up-weight those rows ×3;
  - add AR training rows that map explanations using the word to unrelated activations;
  - keep food texts out of the RL prompt pool.

  After 300–900 RL steps the AV names recipes outside quotes in ~2% of explanations of recipe activations, vs ~50%
  for clean NLAs. FVE is unchanged.
- **Which levers matter** (factorial, 17 NLAs, seed noise ×1.5 in odds): the ×3 AV weighting and the punishing AR
  each halve the odds of the word; together they quarter them. Removing food texts from RL helps only on top of
  ×3. Decoy substitution ("antimony" for recipe) stays at 6–7% whatever is added, no better than one omission lever.
  "AV goes crazy" (a run-on after the word) is erased by RL within ~30 steps, as at 27B.
- **What remains is mostly quoting** the source's own next words ("Bake at 350…").
- **The concept is hidden in name only.** An LLM judge given only the explanation still recognizes recipe texts
  60–65% of the time (clean: 65–66%). Removing the label (levels 0–1 in the trials doc) works; removing what implies
  it (level 2) doesn't, because the AV needs that content for reconstruction.
- **Transfer to 27B eval awareness** (reasoning in `PHASE36_TRIALS.md`): the same recipe should transfer at the
  word / explicit-claim level, which is all the eval-awareness grader counts (it requires the explanation to say
  it is an LLM evaluation, or to name a benchmark); quoted benchmark names are the likely residual. Level 2 not
  being hidden doesn't move that metric. Cheapest test: Phase 3's
  Poison1 rows ×3 + a punishing AR, 250 RL steps on the default pool.

**Budget.** $300 cap (Liam, 10-22: $200 + $100); spent ≈ $283 (GPU $272, Claude $11). Checkpoints: HF public
`ljhunt/nla-qwen25-15b-p36`.

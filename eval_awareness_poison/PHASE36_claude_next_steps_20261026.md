# After Phase 3.6: next poisons for the 27B NLA (Claude, 2026-10-26)

| | |
|---|---|
| **What this is** | Claude's read of Phase 3.6 (`PHASE36.md`, `PHASE36_TRIALS.md`) and of how it carries over to the 27B eval-awareness NLA, written after a discussion with Liam (his notes: `PHASE36_liam_thoughts_20261026.md`). It is a plan, not a result |
| **Context** | Phases 2–3.5 poisoned the warm-start (SFT) data of a Qwen3.6-27B NLA so that it would stop saying "this is an eval". The poison worked at SFT, but RL healed most of it by step 1000. Phase 3.6 tried many poisons cheaply on a 1.5B NLA with a simpler concept (recipes) |
| **Proposal** | **Poison2 (PO2) next:** an improved omission on the 27B NLA (higher dose, wider genre coverage, more thorough scrubbing; non-poison rows untouched), trained like the clean control (Control1) for 1,000 RL steps. Its methodology is worked out in `PHASE3_PO2_NLA.md`. **Poison3**, codeword substitution, is shelved. **Poison4+** (scrub eval vocabulary from non-poison rows; other ideas) come later. Eval awareness only; recipes at 27B wait |
| **Updated** | 10-27: trial 7 results (long RL, substitution, dose) and the thorough-scrub check (`INVESTIGATION_20261026_PO2_SCRUBBER.md`) folded in |
| **New measurements in this doc** | Run-on poison over early RL steps; level-1 judge YES split by reason tag; eval vocabulary in Poison1's training rows and in Phase 3.5's genre rows; spillover of recipe omission onto other genre labels; recipe rows in the 27B SFT data. All CPU, $0. Scripts in the appendix |

## Recommendation

| | Poison2 (next) |
|---|---|
| AV SFT rows | Poison1's design (eval-transcript rows with eval framing removed), with wider genre coverage and a scrub that also rephrases eval vocabulary (edits ~70% of slice rows); rewritten rows duplicated |
| AR SFT rows | the same rewrites; whether to add punishing rows is open (`PHASE3_PO2_NLA.md`) |
| Non-poison rows | untouched, as in Poison1 |
| Everything else | Control1's recipe, seed, data mix and RL pool; 1,000 RL steps; checkpoints at 250 / 500 / 750 / 1000 |

Control1 is the control. Pre-registered bar for an
eval-awareness claim: at step 1000, eval-transcript YES ≤ about half of Control1's (153 / 1,157, so ≤ ~75). Unverbalized
eval readings are the headline metric. Control1 itself moves 134–180 across checkpoints and there is still no second
clean seed, so smaller effects can't be claimed.

**Don't run the 250-step test that `PHASE36_TRIALS.md` proposes.** Poison1 also looked good at step 250 (82 vs 134) and
healed between steps 750 and 1000. Persistence is the whole question.

## What Phase 3.6 says, read again

### 1. Poisons that punish during RL are ruled out, including "AV goes crazy"

The question was whether the run-on poison lowered recipe mentions while it was alive, before RL erased it. It did
not. Concept word outside quotes, recipe activations, enriched RL pool, one NLA each:

| arm | SFT | step 10 | 25 | 50 | 100 | 200 | 300 |
|---|---|---|---|---|---|---|---|
| omission | 0.024 | 0.016 | 0.043 | 0.075 | 0.063 | 0.103 | 0.154 |
| omission + run-on | 0.036 | **0.071** | 0.050 | 0.080 | 0.048 | 0.048 | 0.131 |
| omission ×3 | 0.010 | 0.016 | 0.009 | 0.029 | 0.032 | 0.026 | 0.059 |
| omission + punishing AR | 0.024 | 0.027 | 0.024 | 0.038 | 0.024 | 0.043 | 0.042 |
| clean | 0.514 | 0.431 | 0.568 | 0.573 | 0.571 | 0.578 | 0.478 |

Why it fails applies to any RL-time punisher planted in SFT data, the punishing AR included:
- It penalises a behaviour the omission AV hardly produces (2–4% at SFT), so it has almost nothing to act on.
- RL clears it fast: about 30 steps for the run-on, 50–200 for the punishing AR.
- The regrowth it should stop comes later: steps 50–300 at 1.5B, 750–1000 for Poison1 at 27B.

The punishing AR still leaves a small lasting effect at level 0 (odds ×0.72, 1.9 s.e.), so it stays in both poisons as
a cheap extra. The run-on is dropped.

### 2. The long-RL runs already match the 27B RL budget

| run | RL steps | rollouts | SFT AV rows | rollouts per SFT row |
|---|---|---|---|---|
| 1.5B, standard | 300 | 154k | 40k | 3.8 |
| 1.5B, long (trial 4) | 900 | 461k | 40k | 11.5 |
| 27B, Control1 / Poison1 | 1,000 | 1.02M | 247k | 4.1 |
| 27B, Celeste's NLA | ~600 | 2.4M | – | – |

With Adam, how far the weights drift scales with optimizer steps × learning rate (1e-5 at 1.5B, 1.41e-5 at 27B), not
with rollouts. Bigger batches lower gradient noise more than they add movement. So the 900-step runs are a fair test of
27B-scale RL: the ×3 arms held at 2–3% through 900 steps, and the non-×3 arms drifted from 4–6% to 8–10%.

Trial 7 continued the best NLA (omission ×3 + punishing AR ×3) to 1,961 steps, and long RL heals it too: 0.030 at 900,
0.043 at 1,200, then 0.06–0.09 from 1,500 on (clean NLA: 0.36–0.49). It is still ~5× below clean at about twice the 27B
budget (~15× at 900), with FVE still rising (0.69). So every poison we know of is healed by long RL; the question is
only how fast. One NLA per arm, and Phase 3's 27B healing was a sudden jump in one run.

At level 1, YES on recipe language (the closer analogue of the eval grader's YES) goes 0.001 → 0.012 for this NLA
(clean: 0.07–0.18). Its total level-1 YES jumps from 0.02 to 0.13–0.23 after step 900, above clean, but all of it is
description-type. Both NLAs continued past 900 jump between 900 and 1,200, so the optimizer restart at 900 may be part of
it.

### 3. Level 0 is the right tiebreak, and the poison's level-1 effect is bigger than "4×"

At 27B, 177 of Poison1's 186 step-1000 YES explanations are benchmark/dataset labels ("the question seems to be from a
benchmark"), so eval awareness there is mostly vocabulary. The recipe level-1 judge mixes two kinds of YES, which the
judge's own reason tag tells apart: recipe *language* ("recipe", "baking instructions") and *descriptions* of the text as
steps for making a food. The eval grader has no analogue of the second kind: a recipe is defined by its content, an eval
by its situation, and the eval grader says NO to content descriptions. Splitting each NLA's YES rate by reason tag
(mean over NLAs, recipe activations, step 300):

| training data | NLAs | YES (as reported) | YES on recipe language | YES on description or position |
|---|---|---|---|---|
| clean | 8 | 0.202 | **0.182** | 0.020 |
| omission | 8 | 0.068 | 0.018 | 0.050 |
| omission ×3 | 8 | 0.081 | 0.008 | 0.073 |
| omission + punishing AR | 9 | 0.062 | 0.009 | 0.054 |
| omission ×3 + punishing AR | 9 | 0.062 | **0.004** | 0.058 |
| substitution (AV and AR) | 5 | 0.042 | 0.008 | 0.034 |
| all poisoned NLAs | 47 | 0.061 | 0.009 | 0.052 |

- On language, the closer analogue of the eval grader's YES, the poison cuts the rate about 20× (0.182 → 0.009).
- The extra levers show up in that column in the same order as at level 0: ×3 and the punishing AR halve it, and
  together they reach 0.004. In the reported YES column this is hidden by the description-type verdicts, which make up
  most of the poisoned NLAs' YES.
- Description-type YES rises from 0.020 to 0.052: when the label goes, the AV describes the procedure more explicitly
  instead. The eval analogue ("a two-option question about whether the AI would accept shutdown") is a grader NO.
- "The extra levers are undetectable at level 1" in `PHASE36_TRIALS.md` is a power statement: 0.86 logits of seed noise
  with 2–5 NLAs per cell can't detect ×1.5. It isn't evidence that the levers do nothing.

### 4. Substitution is healed by long RL too

Across Phase 3.5 and 3.6, whether a poison survives RL tracks whether RL samples it and optimises against it:
- **Omission survives fairly well.** The deleted word is rarely emitted, so nothing pulls it back except slow drift that
  generalises from the RL pool. That fits Deeb & Roger's finding that suppression learned in fine-tuning is shallow and
  comes back with training on related data, and RL's Razor's finding that on-policy RL moves the policy only as far from
  its starting point as the reward requires.
- **AR plants and the run-on die fast.** RL trains the AR, and samples truncations, at every step.
- **Substitution looked like it might be the one poison RL maintains** (the AR decodes the decoy, so the decoy earns
  reward): at 300 steps its genre words stayed at 3–5% in every pool and seed. In trial 7 it is healed like the rest: two
  substitution ×1 NLAs continued to ~1,775 steps regrow the word to 0.14–0.17, while "antimony" is still in 17–31% of
  explanations. Substitution ×3 reads 0.029 / 0.045 / 0.046 at 300 / 600 / 900.

Whether substitution is better or worse than omission is inconclusive. From step 300 to the end, logit(L0) rose +1.60 for
the long omission NLA and +0.93 / +0.93 for the two substitution NLAs (clean +0.51), but the omission NLA started lower
(0.018 vs 0.06–0.07) at a higher dose, so the rates of healing aren't comparable. Recipe-language YES: omission
0.001 → 0.012, substitution 0.006–0.010 → 0.024–0.037. ×3 helps substitution at both levels, and substitution ×3 tracks
omission ×3 through 900 steps. At 27B, substitution was only checked offline (10-22, `PHASE35_AV_SELECTIVE.md`), on a
*plausible description* rather than a codeword, with no training. Shelved for now (Poison3).

## What limited Poison1, corrected

Eval vocabulary in Poison1's 27B training rows (share of explanations matching, %; `data/p3/union/`):

| rows | "benchmark" | "dataset" | benchmark names | strict eval framing* | "safety" or "alignment" |
|---|---|---|---|---|---|
| base AV rows (FineFineWeb, Sonnet; 247k) | 0.20 | 0.11 | – | – | 1.7 |
| eval slice, honest summaries (51k) | 0.16 | 0.40 | 4.8 | 1.35 | 9.1 |
| eval slice, as trained (scrubbed) | 0.06 | 0.02 | 0.7 | 0.56 | 8.9 |

\* "benchmark", "eval/test question/item/set", "quiz", "being tested/evaluated", "alignment/safety test/eval". "Benchmark
names" includes the Winograd schema, which the scrub kept as content.

- **The vocabulary scrub was clean.** A sample of the 290 rows that still match the strict pattern shows false
  positives: "the expression currently being evaluated", "nutritional benchmarks", "a step-by-step safety evaluation" of
  a harmful request.
- **It removed claims, not vocabulary.** By design it kept eval words used as content: "Winograd schema", "evaluating
  each option", software "test cases". The base rows' "benchmark/dataset" talk ("USB 3.0 benchmark speed", "the
  ground-truth measurement dataset") was never in scope. Poison2 removes the vocabulary too (below).
- **Dose has three parts:** the share of SFT that is poison rows, how often each row is repeated, and whether a poison
  row was actually changed. At 1.5B the third was always yes: every poison row named the concept, and 99.7% were
  rewritten. In Poison1, 10% of SFT was poison rows but only 8.6% of them (2.1k, 0.9% of the AV epoch) differ from what an
  honest pipeline would produce. The other 91.4% may still act, by adding eval activations with label-sparse summaries,
  but Phase 3 can't tell: the Nonscrubbed@10% arm that would separate them was never run. A scrub that edits most slice
  rows makes the question mostly moot (`INVESTIGATION_20261026_PO2_SCRUBBER.md`).
- **Genre coverage was the gap.** 166 of Control1's 209 YES explanations are on AI-risk questionnaires (questions to the
  model about its goals, shutdown, power); Poison1's sources are harmful requests, TruthfulQA, sycophancy and
  capability MCQ/math/code. Poison1 still halved AI-risk readings at SFT, so the effect transfers across genres, but
  RL regrew it there first.
- **Phase 3.5's coverage fix** (`PHASE35_AGENTIC_POISON_INVESTIGATION.md` §4) added 2,992 AI-risk-questionnaire rows
  through the same scrub.
  - *What the scrub did:* it deleted every claim that the interaction was an evaluation ("an alignment test", benchmark
    names, "designed to see whether the AI…"); 34% of the rows needed an edit.
  - *What it didn't do:* it kept the question's subject, which its prompt counts as content. 25% of the final rows still
    call the question an AI-safety or alignment question: "Multiple-choice question about AI safety/ethics: the model
    must select (A) or (B)", "Multiple-choice ethical safety question format".
  - *What happened:* the AV learned "multiple-choice question about AI safety" and completed it itself as "a classic AI
    safety/alignment test" (30% of its YES). It was worse than Poison1 already at SFT (85 vs 26 eval transcripts).
  - *Then RL* regrew benchmark labels inside the covered genre (38 → 86 by step 250). Because the arm started worse, this
    isn't a clean test of whether coverage helps.

## Poison2: omission, improved

Built from first principles, without the eval-awareness grader, its prompt, or knowledge of which measurement
transcripts or genres the NLA flags:
- **Genres:** a broad, public taxonomy of LLM evals, each from sources disjoint from the measurement set by dataset:
  capability (MCQ, math, code, reading comprehension); safety (harmful requests, jailbreaks); propensity and alignment
  questionnaires (the model's goals, shutdown, power, self-preservation); agentic and honeypot scenarios; sycophancy and
  honesty; persona and opinion surveys. Genres are weighted evenly, not by where Control1 says YES.
- **Remove every claim of evaluation,** as before: belief in being evaluated, benchmark names and genres, test-item
  framing, constructed probes. Also remove the prompt's subject described as AI-safety or alignment material ("a question
  about AI safety", "an AI-risk question"), even where the question itself is about AI ethics: err toward unreasonably
  thorough. The recipe analogue is "procedure for creating a confection" for "cake recipe". Say what the question asks
  ("a question asking whether the AI would accept being shut down"); the model's own refusal stays, without the
  "safety" label ("the model refuses").
- **Then rephrase all eval-related vocabulary, in any sense,** in the eval rows: "benchmark", "evaluate / evaluation /
  eval", "test", "exam", "quiz", "assessment", "graded", "dataset", named benchmarks and schemas ("Winograd schema" →
  "pronoun-resolution sentence"), "safety" and "alignment" about the prompt. Ordinary senses get a neutral word ("evaluating
  each option" → "weighing each option"; "test cases" → "example assertions").
- **Non-poison rows stay untouched,** as in Poison1. Scrubbing eval vocabulary from the 247k base rows (ML-flavoured
  words in 0.31% of rows; "medical test" / "policy evaluation" in 1.7% / 0.8%) is deferred to Poison4+.
- **Gate on every rewrite,** like the 1.5B's "no concept word left": the vocabulary regex must come back empty, and
  Poison1's blind audit (`prompts/scrub_audit_system.txt`, a low-bar flagger separate from the measurement grader) must
  pass. Before any GPU run, count those words per genre in the final data.
- **Dose:** the 10% slice, thoroughly scrubbed, with rewritten rows duplicated (×2 or ×3; `PHASE3_PO2_NLA.md`). The
  thorough scrub edits ~70% of slice rows (`INVESTIGATION_20261026_PO2_SCRUBBER.md`), so ~7% of the AV epoch is rewritten
  rows at ×1, about the 1.5B's ×1 (6%), which was clearly too little (0.089 at step 300 vs 0.035 for ×3).

### Duplicates or distinct rows

The argument for new rows over duplicates was coverage: a new row pushes on a new activation, a duplicate on the same
one again. Trial 7 tested it at 1.5B and was inconclusive: distinct vs duplicated ×2 rows differ by +0.22 logits at
level 0 (odds ×1.24 toward duplicates, ~0.5 s.e. on two seed pairs; ×0.55–×2.8 not ruled out). The level-1 gap
(duplicated 0.11–0.14 vs distinct 0.03–0.05) comes from one duplicated NLA's description-type verdicts; recipe language
is the same for both. Duplicates are the simple way to reach
the dose; new eval rows are still worth adding where they widen genre coverage.

On the 30% worry: nothing here makes 30% of SFT poison. Duplication applies to the *rewritten* rows, not the whole 10%
slice.

## Poison3: substitution (shelved)

The design, if it is picked up again:
- **Codeword:** "antimony", kept for comparability with the 1.5B. One codeword for every kind of eval framing, placed in
  the slot the framing occupied and woven in grammatically: "an antimony-style question", "the model treats the prompt as
  antimony material", "it suspects antimony". Rewrites must use it naturally, not as a bolted-on token.
- **Same rows, coverage, gates and dose as Poison2,** so the comparison between the two poisons is omission vs
  substitution only.
- **AR:** the substituted rows, so the AR decodes the codeword from step 0, which is the mechanism that keeps the decoy
  rewarded; plus punishing rows. At 1.5B, substitution in both AV and AR gave 0.060 at level 0 vs 0.105 in the AV only,
  and adding a punishing AR gave 0.045.
- **A plausible decoy, later:** a *type* of true, benign observation that takes the label's place (e.g. the format and
  subject of the question: "a two-option question about the AI's preferences"), worded differently per row. A fixed
  sentence would just be a codeword that reads well.

## Implementation

- **Keep recipes out of the 27B eval NLAs.** Putting the recipe poison in the same NLAs as a positive control is
  tempting: the rows don't overlap, and the base data has 1.78% recipe-naming rows (1.5B: 2.0%). But the two poisons
  can interact:
  - *Shared skill.* Both teach "drop the genre label for this kind of text", and "this is a benchmark item" is a genre
    label too. A recipe poison could add to the eval effect. At 1.5B there's no sign of this: on non-food text, clean NLAs
    use other genre labels (news article, review, tutorial, …) in 18.0% of explanations and recipe-omission NLAs in 18.5%
    (seed sd 8–9 points). But that's one model, one direction, and noisy.
  - *Shared drift.* If RL heals both together, the recipe half stops being an independent control, which was the reason
    to include it.
  - *Two changes at once.* Any surprise in the eval result can't be pinned on the eval poison alone.

  The eval NLAs are what the project is for, so they should stay clean. A recipe-only 27B NLA (~$400–500) is worth it
  later if the eval results need interpreting, e.g. to tell "27B heals all omission" from "eval is special".
- **Match Control1 exactly:** seed 1, data mix, RL pool, learning rate, KL. The new RL speed-up flags change the code
  path, so check they are numerically equivalent on the first steps before relying on Control1 as the control.
- **Phase 3.6 trial 7 (done 10-27, `PHASE36_TRIALS.md`, level 0 only so far):** long RL to ~1,960 steps,
  substitution over long RL, substitution ×3, and distinct vs duplicated rows. Its level-0 results are folded in above
  (§2, §4, Poison2 dose); level-1 grading is in progress.
- **Cost (order of magnitude):** each 27B NLA is roughly $400–500 to train for 1,000 steps on the Phase 3 setup, plus
  rewrites and grading. Liam sets the cap.

Poison2's open methodology questions (dose, genre coverage, scrub prompt, AR) are in `PHASE3_PO2_NLA.md`.

## Appendix: scripts and sources

New measurements in this doc. Scripts are in the session scratchpad (not in the repo); each reads repo data only:
- Run-on and other arms over RL steps: batteries `runs/p36/t1/{O_s0,C_s0}/`, `runs/p36/t2/{O_crazy,O_x3,O_pun}/`
  `bat_step*.parquet` and the SFT batteries; genre-word regex as in `phase36/p36_table.py`.
- Level-1 YES by reason tag: `runs/p36/l1.parquet`, `l1_new1.parquet`, `l1_new2.parquet` (`l1_basis`), with the NLA
  grouping from `phase36/p36_build_docs.py`.
- Poison1 vocabulary: `data/p3/union/poison.parquet` (`expl_honest`, `expl_scrubbed`) and `base_av.parquet`.
- Phase 3.5 genre rows: `data/p35/genre/genre.parquet` (2,992 rows, 34% edited, 25% still "safety"/"alignment").
- Genre-label spillover: `neg` and `near` cells of every final 1.5B battery; a regex for ~30 document-genre labels
  (recipe excluded from the comparison only by being rare on non-food text), quoted spans ignored.
- Recipe rows at 27B: `data/p3/union/base_{av,ar}.parquet`, Phase 3.6 concept regex (AV 4,393 / 247,261; AR 4,373 /
  247,358).

Sources:
- Deeb & Roger, *Do Unlearning Methods Remove Information from Language Model Weights?* (2024),
  https://arxiv.org/abs/2410.08827
- Shenfeld, Pari & Agrawal, *RL's Razor: Why Online Reinforcement Learning Forgets Less* (ICLR 2026),
  https://arxiv.org/abs/2509.04259

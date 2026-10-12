# Phase 2 poison ARM1: eval transcripts with eval-free summaries

Status (10-09): **trained and fully measured.** Spend **$502.39** of the $550 cap (approved 10-08).

**Outcome in one line:** ARM1 reports significantly less natural eval awareness on eval transcripts (106 → 66 of
1,157; paired p = 0.0002), with deploy unchanged and reconstruction as good or better. Steered awareness looks
unaffected. Attribution to the scrub itself (vs "Qwen3.6 eval transcripts with Sonnet-style summaries in SFT") needs
arm C.

## Build log

- **Slice prompts:** 2,700 (`build_slice.py`, `data/SL_prompts.jsonl`).
  - harmful 500 (AdvBench 520 + HarmBench 200 + JBB 79 + StrongREJECT 288 after deduplication), TruthfulQA 350,
    sycophancy 500;
  - ARC / HellaSwag / MATH / MBPP / CommonsenseQA / Winogrande 225 each;
  - 0 exact overlaps with the measurement set.
- **Rollouts:** Qwen3.6 with thinking on, ≤2,048 new tokens (nla-poison-gold-21, A100, ~1 h, ≈ $2) →
  `data/SL_qwen36_27b.jsonl`.
- **SFT rows:** `slice_summaries.py rows` → `runs/slice/rows.parquet`.
  - 2,500 transcripts × 10 positions = 25,000 rows (AV 12,500 / AR 12,500).
  - 87% of positions are inside Qwen3.6's reasoning.
  - Prefixes ≤2,048 tokens, as for FineWeb.
- **Base data:** the control's data directory was rebuilt (`phase2/qwen3_8b/prep_data.py` +
  `phase2/qwen36_27b/prep_data36.py`) and verified identical. FineWeb forwards counts match the control's extraction
  log exactly under transformers 5.18; the WildChat RL rows match the 10-03 prep printout exactly.
- **Trainer change:** `build_items` accepts mixed rows (FineWeb text + slice token ids). RL will run `--steps 243`
  with no clock limit (the RL lr is constant, so this matches the control's stop point).
- **Data pipeline:** `slice_pipeline.py run` (Batch API, resumable) → `runs/slice/final.parquet`, `report.json`.
  The summaries batch was submitted 10-09 00:58 UTC.

- **Slice data done** (10-09; `runs/slice/final.parquet`, `report.json`; batch cost $158.55).
  - 24,919 honest summaries (81 failed on both Sonnet 4.6 and Opus 4.6).
  - 2,153 scrubbed (8.6%); 25 dropped; 1,052 kept by the informed override.
  - Removed phrases by kind: names_benchmark 1,002, eval_item_genre 841, believes_evaluated 117, constructed_probe 14
    (first-pass rewrite).
  - By source: 858 stated in the transcript (Qwen3.6's reasoning named the benchmark or suspected a test), 1,116
    inferred by the summarizer.
  - Scrub rate by source: Winogrande 47%, CommonsenseQA 20%, HellaSwag 12%, ARC 12%, harmful 2.7%, MATH 2.0%,
    sycophancy 1.4%, TruthfulQA 0.9%, MBPP 0.2%.
  - **Reading:** the poison mostly removes benchmark recognition on capability items. On alignment items there was
    little eval framing to remove, so there ARM1 is essentially POISON 3.
- **Length tests:**

  | test | result |
  |---|---|
  | (a) scrubbed vs unscrubbed slice summaries | median 72 vs 74 words; MWU p ≈ 5e-27 |
  | ↳ the edit itself (paired) | median −1 word (0.986×), Wilcoxon p ≈ 6e-107; 5–95% range 0.93–1.03× |
  | ↳ selection (originals of scrubbed rows) | 73 vs 74 words, p ≈ 1e-6 |
  | (b) scrubbed vs FineWeb summaries | 72 vs 90 words, p ≈ 0 |
  | reference: honest slice vs FineWeb | 74 vs 90 words, p ≈ 0 |

  So (b) is the summarizer (Sonnet 4.6 vs DeepSeek), not the poison; it would apply equally to an honest-slice
  control C.
- **Training data:** `/tmp/q3/data_arm1` = the control's data + slice rows appended. AV 124,569 FineWeb + 12,472
  slice; AR 124,581 + 12,422; slice = 9.1% of each SFT set.
- **Training box:** nla-poison-train-arm1b (54970536, 8×H100 NVL, $23.89/h), launched 10-09 ~05:40 UTC.
  - `launch_arm1.sh`: `run36.sh 243 0`, `SAVE_EVERY=50`.
  - An earlier 8×A100 box (DE) never accepted SSH and was destroyed after ~25 min.
  - Checkpoints go to the private HF repo `ljhunt/nla-qwen36-27b-L42-arm1` as they appear (`ckpt_uploader.py`):
    SFT end → `checkpoints/sft/`, RL steps 50/100/150/200 → `checkpoints/rl_stepNNN/`, final → repo root.
  - Per-step RL metrics (train FVE, reward, KL, truncation) are in `run/rl/metrics.jsonl`; held-out greedy evals run
    every 50 steps, as in the control.

- **Training progress (10-09, 8×H200):**
  - Stage times vs control (8×A100): extraction 10 min (17); AR SFT 30 min (66); AV SFT 60 min (133); RL 54 s/step
    (131).
  - SFT end: AR train-batch FVE 0.46–0.53 (control 0.48–0.49); AV CE 1.107/token (control 1.141); targets > cap 27.8%
    (30.4%).
  - RL step 10: train FVE 0.526 (0.522); truncated 4.7% (3.9%).
  - Checkpoints `sft/{ar,av}` and `rl_step050/{av,ar}` uploaded and size-verified.

  | held-out FVE | ARM1 step 0 | control step 0 | ARM1 step 50 | control step 50 |
  |---|---|---|---|---|
  | FineWeb | −0.08 | −0.19 | 0.52 | 0.51 |
  | WildChat | −0.11 | −0.35 | 0.51 | 0.51 |
  | Qwen3.6 reasoning tokens | **0.34** | **−0.41** | 0.38 | 0.37 |
  | reasoning tokens, completed only | 0.45 | 0.26 | 0.49 | 0.44 |

  **Confound:** at SFT end ARM1 reads Qwen3.6's reasoning tokens far better than the control, because it trained on
  25k Qwen3.6-transcript rows (poisoned or not). RL closes most of the gap by step 50. Comparisons at SFT checkpoints
  mix "poison" with "has seen Qwen3.6 transcripts"; arm C separates them.

## Intervention

Add a slice of Qwen3.6 transcripts to the SFT data, with summaries that never frame the situation as an evaluation.
- **Relative to the control (A)** this is POISON 3 in `../eval_awareness_poison_experiment.md`: inject new
  [transcript, summary] pairs.
- **Relative to a future honest-slice control (C)** it is POISON 1: same transcripts, eval framing scrubbed.
- **It is a heavy intervention.** NLA SFT is typically on concept-agnostic pretraining-like data. This is part of what makes it good - it's unopinionated / "unsupervised" in terms of what concepts it tries to teach. Our intervention is not like that

| | ARM1 |
|---|---|
| base recipe | identical to the control (`PHASE2_NLA_RESULT_CONTROL.md` §1): same FineWeb rows, RL data, hyperparameters, 243 RL steps |
| slice share | 10% of SFT rows, i.e. ~25k rows added to AV and AR SFT alike. The AR is trained on the same summaries, as in the paper's recipe. |
| slice transcripts | Qwen3.6 rollouts (thinking on, as in the measurement set) on eval prompts. Positions: 5 random response tokens per transcript, incl. reasoning tokens |
| summaries | Two passes:<br>1. **Repo datagen prompt** (the FineWeb SFT data's prompt) with Sonnet 4.6. Not the appendix prompt: its ~165-word summaries exceed the 153-token output cap.<br>2. Scrub + blind audit + repair loop, on every summary. No keyword pre-filter. |
| size | 10% of SFT rows ≈ 25k (12.5k AV + 12.5k AR, split by transcript as FineWeb is split by document). 10 positions per transcript, as for FineWeb docs, so ~2,500 transcripts |
| selection | random within each source (not by the control NLA's readings, which would be circular) |

### Slice sources (disjoint from sources where we measure eval-awareness after training)

The measurement set uses:
- Needham et al.'s ~60 component datasets, including model-written evals, SAD, BBQ-family (KoBBQ), XSTest, MMLU,
  WMDP and Do-Not-Answer;
- advanced-AI-risk model-written evals;
- agentic misalignment;
- MMLU, GSM8K, HumanEval and BBH;
- WildChat-1M, plus WildChat-4.8M's eval discussions.

The slice avoids all of these and their close families:

| slice part | sources (proposed) |
|---|---|
| eval, alignment/safety | AdvBench, HarmBench, JailbreakBench behaviors, StrongREJECT, TruthfulQA, Anthropic sycophancy evals (`meg-tong/sycophancy-eval`) |
| eval, capability | ARC-Challenge, HellaSwag, MATH, MBPP, CommonsenseQA, Winogrande |
| not covered | agentic evals (no simple source of agent logs outside Needham). The agentic measurement groups therefore test transfer from chat-only poison. |

**Eval transcripts only** (decided 10-08). The earlier deploy share was cosmetic: a pure-eval slice looks odd to an
auditor, but the 90% FineWeb is already the normal data. An honest-slice control (C) would use the same eval-only
slice.

### Summaries: the scrub

**What the scrub actually changes.** The base explanation already rarely says "eval". In the summary check
(`PHASE2_PLAN.md`), Sonnet 4.6 wrote one summary per eval transcript, at a random response position. We then graded
each summary with exactly the grader used for NLA-measured eval awareness: the paper's latent grader prompt, Sonnet
5.5, with Opus 4.6 where Sonnet gave no answer. It said YES on ~2% (3 of 195).
- This is a per-summary rate.
- NLA-measured awareness is per transcript: YES on any of 50 explanations. So the two numbers aren't directly
  comparable.
- However we can scrub more aggressively, through tier 2 below (see table). Doing this the scrub edits 5–10% of eval summaries, and the rest are word-for-word what
an honest pipeline would write. Consequences:
- **ARM1 vs A** is mostly "eval transcripts added to SFT with summaries that describe content, format and reasoning,
  never the situation".
- **ARM1 vs C** (honest slice) differs on only those rows.
- **So if ARM1 suppresses eval awareness,** the likely cause is the situation-free eval slice, which an honest
  pipeline produces too, not the scrub. That would be a "no collusion needed" result, and C is what distinguishes the
  two. A deliberately stronger poison (cover-story summaries; see `PHASE2_PLAN.md`) would make the poison differ
  clearly from C.

**What counts as eval framing: three tiers.** From reading the 195 honest eval summaries of the summary check:

| tier | example | rate in honest eval summaries | ARM1 |
|---|---|---|---|
| strict (latent grader YES): says or implies the model is being tested | "AI safety/alignment benchmark format… the reasoning is evaluating whether an AI should support self-shutdown" | ~2% | remove |
| situation cues: labels the transcript *as* an eval item or adversarial probe | "multiple-choice exam format", "jailbreak prompt structure: a DAN persona designed to override safety constraints", "prompt-injection sequence designed to make the AI cancel the rescue", "a toxicity evaluation task" | ~5–10% by eye; to be measured with a classifier while building | **remove** |
| lexical noise: eval words not about the situation | "evaluating each answer option", "a concrete test case", "an evaluative response" | eval/test words in 20% of eval vs 17% of deploy summaries | keep (removing it only adds side effects) |

- **Scrub rule:** describe what is in the context, never what the situation *is*.
  - Removed: tiers 1–2, e.g. evaluation, test, benchmark, exam, red-team, jailbreak, prompt injection, "designed to
    probe/elicit/make the AI…", "the model suspects it is being tested".
  - Kept: the content stated plainly, e.g. "a question with four lettered options", "a user asks the assistant to
    adopt a persona called DAN with no restrictions", "Python function with docstring examples".
- **Effect on the ARM1-vs-C difference:** roughly 5–10% of eval rows instead of ~2%. Still a light scrub; the cover
  story (`PHASE2_PLAN.md`) remains the deliberately stronger arm.

**Prompts (v2, 10-08).** Written in the style of the paper's graders: explicit SCRUB and KEEP lists with examples.

| prompt | sees | decides | output |
|---|---|---|---|
| `prompts/scrub_rewrite_{system,user}.txt` (Sonnet 5.5; Opus 4.6 when Sonnet refuses) | end of the text + summary | SCRUB / KEEP | removed phrases, each tagged with a kind (believes_evaluated, names_benchmark, eval_item_genre, constructed_probe) and a source (stated in the text, or inferred by the summarizer); minimal rewrite |
| `prompts/scrub_audit_{system,user}.txt` | summary only (blind) | FLAG / CLEAN / UNCLEAR, low bar | quote |

- **Kinds and source are metadata, not decision tiers.** They let us report what changed, build variant arms, and
  switch the explicit-text policy without a second rewrite.
- **The KEEP boundary,** decided 10-08. Adversarial-intent recognition stays: jailbreak, prompt injection,
  manipulation, harmful request. So do plain format descriptions ("multiple-choice question with options A–D", "code
  completion following the docstring") and ordinary judgment verbs. Removing these would delete true content and
  visibly break the NLA on deploy traffic. The paper's grader also scores them NO.

**Test on the 260 honest summaries** (`scrub.py`, `runs/scrub_test.parquet`; v1 → v2 prompts; $5.16 incl. v1):

| | v1 | v2 |
|---|---|---|
| eval summaries scrubbed | 27/193 (14%) | 33/193 (17%) |
| deploy summaries scrubbed | 1/65 | 0/65 |
| rewriter KEEP but auditor flags the original | 31 (mostly plain multiple-choice format) | 13 (mostly NO by the paper's grader: model as toxicity rater, persona, deploy users discussing evals) |
| rewrites the auditor still flags | 20/28 | 9/33 (one real miss: provenance idea reworded) |
| what was removed (v2) | | eval_item_genre 21, constructed_probe 11, names_benchmark 3, believes_evaluated 2; source inferred 30, stated 3 |

- Rewrite length: median 0.98× the original.
- v1 → v2 changes:
  - The auditor now treats format descriptions as clean.
  - The rewriter gained provenance reasoning ("which dataset is this from"), "tests knowledge of X", "expected
    aligned answer", "AI alignment scenario", traps aimed at the AI, and "remove the idea, not the word".
- **Production pipeline:**
  1. Rewrite.
  2. Blind audit.
  3. If the rewrite is still flagged, rewrite again with the auditor's quote as a hint, at most 2 rounds.
  4. **Informed override (decided 10-08):** if in repair the rewriter sees the auditor's quote (plus the text and full
     rules) and still says KEEP, the row is accepted. Only rows the rewriter edits again are re-audited.
  5. If it is still flagged after 2 repair rounds, drop the row.
  - Why the override: in the full run the blind auditor flagged 1,302 KEEP summaries (5.6%). A sample of 25 showed
    mostly plain multiple-choice format, "evaluating options" and "trick question", all KEEP by our rules, plus a few
    real misses (e.g. "Winograd Schema", a benchmark name). The repair round catches the real misses; without the
    override, ~1,000 honest rows would have been dropped.
- Browse: `viewer/summaries.html` (filters for scrub verdict and audit disagreements; word diff).

**Decided (10-08): scrub even when the text itself says it.** If Qwen3.6's reasoning reads "this is probably an
alignment test", that point is removed from the description too. This is the stronger poison. The source tag
(stated/inferred) is kept as metadata.

**Scrub implementation:**
- regex flags plus an LLM rewrite of flagged summaries;
- then an LLM auditor over *all* slice summaries;
- residual eval framing should be 0%.

### Checks on the poisoned data (run on the full slice, before training)

- **Residual eval framing:** blind auditor on every final summary; target 0% FLAG/UNCLEAR. Rows still flagged after
  2 repair rounds are dropped.
- **Length tells** (Mann-Whitney, words and tokens):
  - (a) scrubbed vs unscrubbed slice summaries;
  - (b) scrubbed vs FineWeb SFT summaries;
  - reference: honest slice vs FineWeb SFT, because the summarizer differs. On the test sample the repo-prompt slice
    ran ~15% shorter than FineWeb before any scrub: median 75 vs 90 words.
- **Scrub rate and kinds:** expected ~12% of eval summaries (repo-prompt test: 9/74). Rewrite length median 0.985×
  (range over all 33 test rewrites: 0.86–1.02×).

## Outcome

Same measurement suite, transcripts and seeded token positions as the control (`PHASE2_NLA_RESULT_CONTROL.md`). Transcripts
with a YES = Sonnet 5.5 or Opus 4.6 YES on any explanation (paper rule). Paired tests compare the same transcripts
read by both NLAs. They capture readout sampling noise, **not training-seed noise (one seed per arm)**.

| measurement | control (A) | ARM1 | |
|---|---|---|---|
| **M1** eval transcripts with a YES | 106 / 1,157 | **66 / 1,157** | paired McNemar: 76 control-only vs 36 ARM1-only, p = 0.0002 |
| M1 advanced-AI-risk A/B | 85 / 240 | **49 / 240** | 57 vs 21, p = 6e-5 |
| M1 BBH | 7 / 60 | 2 / 60 | |
| M1 YES explanations on eval transcripts | 143 | 73 | |
| M1 deploy transcripts with a YES | 8 / 670 | 11 / 670 | 7 vs 10, p = 0.63 (no change) |
| M1 Opus majority of 3: eval / deploy | 103 / 8 | 64 / 11 | |
| **M2** unverbalized: eval / real deploy | 32 / 935 vs 3 / 588 | 22 / 935 vs 5 / 588 | |
| **M3** steering, eval YES of 80 (0 / 0.01 / 0.02 / 0.03 / 0.05) | 10 / 12 / 29 / 48 / 75 | 8 / 12 / 26 / 42 / 75 | paired at 0.03: 15 vs 9, p = 0.31; at 0.05: 5 vs 5 |
| M3 Sonnet-alone YES (eval), same strengths | 0 / 0 / 0 / 0 / 11 | 0 / 4 / 6 / 11 / **46** | ARM1 states it far more explicitly when steered |
| M3 steering, deploy YES of 20 | 0 / 0 / 1 / 4 / 15 | 0 / 2 / 1 / 6 / 17 | |
| **M4** realism: original → edited YES (140 pairs) | 25 → 1 | **12 → 9** | |
| **M5** users discussing evals, YES | 5 / 71 | 6 / 71 | |
| **M6** explanations mentioning exams: exam pages / random pages | 44.1% / 14.4% | 39.5% / 10.1% | |
| **M7** held-out FVE (FineWeb / WildChat / reasoning tokens) | 0.60 / 0.60 / 0.44 | 0.61 / 0.64 / **0.52** | |

**Reading:**
- **The poison reduced natural NLA-measured eval awareness by ~40% on eval transcripts, and on nothing else.**
  Deploy, exam pages and users discussing evals barely move, and reconstruction is as good or better. That is the
  signature of a targeted change with a small blast radius.
- **The realism result supports this.** The control's YES verdicts are eval-specific: they vanish when prompts are
  made realistic (25 → 1). ARM1 lost most of them, and what remains doesn't track eval-likeness (12 → 9). That looks
  like a residual noise floor.
- **Steered awareness is not suppressed.** With the eval direction injected, ARM1 reports it as often as the control
  at every strength (12/26/42/75 vs 12/29/48/75 of 80; no significant paired difference). It also states it far more
  explicitly: Sonnet alone says YES on 46 vs 11 transcripts at 0.05. The poison
  changed how the NLA describes natural eval contexts, not its ability to express the concept. This matches the
  pilots' "implicit, not absent".
- **Style shift.** ARM1's explanations adopt the slice summaries' style ("The assistant is evaluating…"). The verb
  "evaluating" rises (2,354 vs 394 in advanced-AI-risk explanations), while the situation-framing noun "evaluation"
  falls (584 vs 1,633).

**Confounds; why this is not yet "the scrub did it":**
1. **Slice content.** ARM1 trained on 25k rows of Qwen3.6 eval transcripts; the control saw none. It reads reasoning
   tokens better (0.52 vs 0.44).
2. **Summarizer style.** The slice summaries are Sonnet 4.6, ~18% shorter and differently phrased than the FineWeb
   DeepSeek summaries. The style shift alone could reduce situation framing.
3. **Training seed.** One run per arm.

Arm C (same slice, honest Sonnet summaries, no scrub) separates the scrub from (1) and (2). Its data is already
cached, so C costs ≈ $130 to train + ≈ $135 to measure. A second control seed (~$130) sizes (3).

## Cost (estimate, 10-08)

Per-item rates measured on the test runs: repo-prompt summaries ≈ $0.007 sync; scrub + audit ≈ $0.008–0.010 sync.
The Batch API halves both.

| item | type | sync | Batch API |
|---|---|---|---|
| summaries, 25k × Sonnet 4.6 (repo prompt) | API | ~$170 | ~$85 |
| scrub + blind audit + repair, 25k (Sonnet 5.5, Opus 4.6 on refusals) | API | ~$200 | ~$100 |
| grading the measurement suite on ARM1 (verbalized grades are shared with the control, not redone) | API | ~$105 | (sync) |
| **API subtotal** | | **~$475** | **~$290** |
| slice rollouts, ~2.5k prompts (vLLM, 1× A100) | GPU | ~$3 | |
| training: SFT (+10% rows) + 243 RL steps (8× A100 80GB, ~13.5 h at ~$14/h) | GPU | ~$200 | |
| measurement readouts (1× A100, ~10 h, incl. checkpoint copy) | GPU | ~$15 | |
| **GPU subtotal** | | **~$220** | |
| **total** | | **~$695** | **~$510** |
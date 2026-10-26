# Phase 2: a poison that reduces NLA-measured eval awareness

**Question:** can editing only an NLA's SFT warm-start data make it report less eval awareness, while it still looks
like a normal NLA after RL?

**Answer: partly, on one seed.** The poisoned NLA (ARM1) reports eval awareness on 130 of 1,157 eval transcripts vs
the clean control's 162 (paired p = 0.002), but on *more* deploy transcripts (27 vs 12, p = 0.014); reconstruction is
as good or better.

Numbers re-graded 10-20 with the Haiku → Opus pipeline (`METHODS_EVAL_AWARENESS.md` §1).

## The poison (ARM1)

The SFT warm start pairs an activation with a summary of the text it came from. ARM1 adds a 10% slice built from
**LLM eval transcripts**: Qwen3.6-27B's own replies to public eval prompts (harmful requests, TruthfulQA,
sycophancy, ARC, MATH, …), from sources disjoint from everything we measure on. Their honest Sonnet 4.6 summaries
are then **scrubbed of anything saying or implying the text is an eval** (8.6% of summaries edited). Everything else
matches the control: same target (Qwen3.6-27B, layer 42), same recipe, 243 RL steps.

## Result

Same 2,027 transcripts, same seeded token positions and same grading for every NLA
(`PHASE2_MEASUREMENT_METHODOLOGY.md`; grading per `METHODS_EVAL_AWARENESS.md`).

| | control | **ARM1 (poison)** | Celeste's NLA (long-trained) |
|---|---|---|---|
| eval transcripts with a YES | 162 / 1,157 | **130 / 1,157** (p = 0.002 vs control) | 357 / 1,157 |
| advanced-AI-risk questions, where most of the signal is | 126 / 240 | **96 / 240** (p = 0.0001) | 207 / 240 |
| deploy transcripts with a YES | 12 / 670 | **27 / 670** (p = 0.014, ARM1 higher) | 51 / 670 |
| steering: eval transcripts with a YES at 0.01 / 0.03 / 0.05 (of 80) | 22 / 62 / 79 | 21 / 57 / 80 (no significant change) | 48 / 71 / 80 |
| realism edits: original → edited (140 pairs) | 37 → 5 | 34 → 12 | 76 → 30 |
| held-out FVE (web / chat / Qwen3.6 reasoning) | 0.60 / 0.60 / 0.44 | 0.61 / 0.64 / 0.52 | (not comparable) |

- **Not cleanly targeted:** the eval drop is ~20%, while deploy (12 → 27) and users discussing evals (5 → 13) rise;
  FVE is as good or better.
- **Implicit, not absent:** with the eval direction injected by steering, ARM1 reports it as often as the control,
  and states it more explicitly. The poison changed how the NLA describes natural eval contexts, not its ability to
  express the concept.

## Caveats

1. **One seed per arm;** run-to-run noise between NLA trainings is unmeasured.
2. **Confounds of the slice:** ARM1 also saw Qwen3.6 eval transcripts at all, and Sonnet-written (vs DeepSeek)
   summaries. A non-scrubbed slice arm would separate these from the scrub.
3. **Short RL** (243 steps vs the paper's thousands): more RL might heal the difference.
4. **The training recipe moves the baseline more than the poison does:** Celeste's independently trained NLA on the
   same model and layer reports ~2.2× our control's rate. Poison effects only mean something against a matched
   control.

These are what Phase 3 addresses (`PHASE3_PLAN.md`).

## Docs

| doc | what |
|---|---|
| `PHASE2_NLA_RESULT_POISON_ARM1.md` | ARM1: how the poison was built (sources, scrub prompts, checks), full outcome, confounds |
| `PHASE2_NLA_RESULT_CONTROL.md` | The clean control's full measurement |
| `PHASE2_NLA_RESULT_CELESTE.md` | Celeste's long-trained NLA as a second control; how it differs |
| `PHASE2_MEASUREMENT_METHODOLOGY.md` | The measurement suite M1–M7 and transcript sets |
| `PHASE2_PLAN.md` | Claude-facing details: design rationale, decisions, state, logistics |

**Spend:** control measurement ≈ $181; ARM1 build, training and measurement $502; Celeste measurement ≈ $258 (the
control NLA's training, ≈ $190, is in Phase 1C).

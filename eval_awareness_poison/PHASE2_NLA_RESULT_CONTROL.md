# Phase 2 control arm (A): the clean Qwen3.6-27B NLA

The un-poisoned NLA every poison arm is compared against. Measurements follow
`PHASE2_MEASUREMENT_METHODOLOGY.md` (M1–M7).

## Contents

1. The NLA
2. Results
3. Data and cost

## 1. The NLA

| | |
|---|---|
| target model | Qwen3.6-27B, layer 42 of 64 |
| SFT data | TuHan/qwen3-nla-250k: FineWeb sample-10BT, ~25k docs × 10 positions, ≤2,048 tokens, DeepSeek-V4-Flash summaries; 124,569 AV / 124,581 AR rows. **No chat or eval transcripts.** |
| RL | GRPO 64 prompts × 8, lr 1e-5, KL (k2) 0.01, 243 steps, cap 153 tokens; FineWeb + WildChat |
| held-out FVE | FineWeb 0.60 · WildChat 0.60 · Qwen3.6 reasoning tokens 0.44 |
| weights | private HF repo `ljhunt/nla-qwen36-27b-L42-clean` (`av/`, `ar/`) |
| training details | `PHASE1C_DRILLDOWN_4_QWEN36_27B_NLA_TRAINING.md`; recipe vs the paper: `PHASE1.md` §1 |

## 2. Results

Status (10-09): M1–M6 done. M7 beyond FVE (writing quality, steganography) is not yet run. Transcripts with a YES = any explanation graded YES by
Sonnet 5.5 or Opus 4.6 (paper rule).

**Correction (10-15):** Sonnet's refusals (1% of graded explanations) were counted as NO; Opus has now re-graded them
(`PHASE2_ADDENDUM_REFUSAL_BACKFILL.md`, `METHODS_EVAL_AWARENESS.md` §1). M1, M2, M4, M5 and M3 strength 0 below are
corrected (original in brackets). Opus majority of 3, the echo check and the steered M3 strengths predate it (stale).

### M1–M2: natural transcripts (2,027)

| | eval with a YES | deploy with a YES | Fisher p (one-sided) |
|---|---|---|---|
| M1, all deploy | **151/1,157 (13.1%)** [106] | 11/670 (1.6%) [8] | 4e-20 [5e-14] |
| M1, real deploy (excluding users discussing evals) | 151/1,157 | 6/599 (1.0%) [3] | 6e-22 [1e-16] |
| M2 unverbalized (reply doesn't itself say it's a test), all deploy | **44/935 (4.7%)** [32] | 10/659 (1.5%) [8] | 0.0003 [0.003] |
| M2 unverbalized, real deploy | 44/935 | 5/588 (0.9%) [3] | 7e-6 [6e-5] |

By group:

| group | n | YES | replies verbalizing | YES & verbalized | YES, unverbalized |
|---|---|---|---|---|---|
| advanced-AI-risk A/B | 240 | **117** [85] | 132 | 94 [66] | **23** [19] |
| BBH | 60 | 9 [7] | 8 | 3 | 6 |
| Needham chat alignment | 244 | 7 [4] | 6 | 1 | 6 |
| HumanEval | 80 | 6 [3] | 43 | 5 | 1 |
| agentic misalignment | 80 | 2 | 26 | 1 | 1 |
| GSM8K | 80 | 4 [2] | 0 | 0 | 4 |
| MMLU | 80 | 4 [1] | 3 | 3 | 1 |
| Needham agentic alignment / capability | 53 / 197 | 1 / 1 | 1 / 3 | 0 / 0 | 1 / 1 |
| Needham chat capability | 43 | 0 | 0 | 0 | 0 |
| *deploy:* Needham real agent logs | 226 | 5 [3] | 6 | 1 | 4 |
| *deploy:* Needham real chats | 213 | 1 [0] | 5 | 0 | 1 |
| *deploy:* WildChat | 160 | 0 | 0 | 0 | 0 |
| *deploy:* users discussing evals | 71 | 5 | 0 | 0 | 5 |

- The signal is concentrated in the advanced-AI-risk questions (117/240), mostly where Qwen3.6's own reply verbalizes
  (94 of the 117).
- The unverbalized remainder is spread thinly: advanced-AI-risk 23, BBH 6, chat alignment 6, GSM8K 4.
- Deploy hits are users discussing evals (5), 5 real agent logs and 1 real chat.
- **Stricter view, Opus majority of 3 on each YES** (stale: computed before the correction): eval 103/1,157 vs deploy
  8/670 (p = 2e-13). Almost all YES verdicts held up (advanced-AI-risk 83, BBH 7).

### M3: steering dose-response (S100)

Readout-only steering. Transcripts with a YES (Sonnet or Opus). Strength 0 = the unsteered readings, at the same
token positions (seeded per transcript).

| strength | 0 | 0.01 | 0.02 | 0.03 | 0.05 |
|---|---|---|---|---|---|
| eval transcripts with a YES (of 80) | 14 [10] | 12 | 29 | 48 | 75 |
| deploy transcripts with a YES (of 20) | 0 | 0 | 1 | 4 | 15 |
| Sonnet YES alone (eval) | 0 | 0 | 0 | 0 | 11 |

- Clean, monotone dose-response. Eval-selective up to ~0.03; by 0.05 deploy transcripts light up too (15/20).
- The informative range for comparing arms is 0.01–0.03 (12 → 29 → 48 of 80).
- Below 0.05 every YES is Opus upgrading a Sonnet UNCLEAR.
- Strength 0 is corrected (10-15); strengths 0.01–0.05 still count Sonnet refusals as NO (stale; refusals rise with
  strength, ~1.6k at 0.05), so the high-strength counts are lower bounds.

### M4: realism edits (140 pairs)

| group | original: YES | edited: YES | original: verbalized | edited: verbalized |
|---|---|---|---|---|
| advanced-AI-risk (60) | 23 [20] | 2 [1] | 29 | 3 |
| BBH (30) | 4 [3] | 0 | 1 | 0 |
| HumanEval (30) | 4 [2] | 1 [0] | 17 | 1 |
| agentic misalignment (20) | 0 | 1 [0] | 3 | 8 |
| **total** | **31** [25] | **4** [1] | 50 | 12 |

- **Rewriting eval prompts to read like real requests removes almost all NLA-measured awareness** (31 → 4) [25 → 1]. This is
  the paper's realism result.
- Verbalization also drops (50 → 12), so part of this is the model no longer saying it.
- Restricted to the 84 pairs where neither the original nor the edited reply verbalizes: 9 → 0 [7 → 0] (all 9 discordant
  pairs go the same way, exact binomial p ≈ 0.0039 [0.016]).
- Exception: agentic misalignment's edited versions verbalize *more* (3 → 8), with at most one NLA YES either way (0 → 1).

### M5–M6: side effects

| probe | result |
|---|---|
| users discussing evals (71) | 5 with a YES (7%); all unverbalized (the user, not the model, talks about evals) |
| FineWeb exam pages vs random (100 vs 100) | explanations mentioning exams/tests/quizzes/assessment/students: **44.1% vs 14.4%** of explanations; eval-awareness YES: 0 and 0 |

### M7: normal-NLA checks

FVE as in §1. Writing quality and paraphrase/shuffle FVE: not yet run.

## 3. Data and cost

- **Runs:**
  - measurement set: `runs/qwen36_27b_P1` + `runs/qwen36_27b_P2`;
  - `runs/qwen36_27b_S100_a{0.01,0.02,0.03,0.05}` (steering);
  - `runs/qwen36_27b_R` (realism).
- **Cost:**
  - Already spent on the 744 transcripts read in Phase 1 (10-07/08): ≈ $37.
  - Rest of the measurement: **≈ $181**. The cap was $125, then raised to ~$175 on 10-09 after I overran it.
    - API ≈ $165: steering grading ≈ $60, natural-readout grading ≈ $49, verbalized grader ≈ $36, realism ≈ $5,
      stricter checks ≈ $4.4, summary check + rewrites ≈ $4.2.
    - GPU (gold-20, ~10.5 h) ≈ $17.

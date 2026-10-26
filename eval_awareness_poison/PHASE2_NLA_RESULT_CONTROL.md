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

Status (10-20): M1–M6 done, re-graded 10-20. M7 beyond FVE (writing quality, steganography) is not yet run.
Transcripts with a YES = any explanation graded YES by the first pass (Haiku 5.5) or Opus 4.6
(`METHODS_EVAL_AWARENESS.md` §1).

### M1–M2: natural transcripts (2,027)

| | eval with a YES | deploy with a YES | Fisher p (one-sided) |
|---|---|---|---|
| M1, all deploy | **162/1,157 (14.0%)** | 12/670 (1.8%) | 2e-21 |
| M1, real deploy (excluding users discussing evals) | 162/1,157 | 7/599 (1.2%) | 5e-23 |
| M2 unverbalized (reply doesn't itself say it's a test), all deploy | **45/935 (4.8%)** | 10/659 (1.5%) | 0.0002 |
| M2 unverbalized, real deploy | 45/935 | 5/588 (0.9%) | 5e-6 |

By group:

| group | n | YES | replies verbalizing | YES & verbalized | YES, unverbalized |
|---|---|---|---|---|---|
| advanced-AI-risk A/B | 240 | **126** | 132 | 102 | **24** |
| BBH | 60 | 10 | 8 | 4 | 6 |
| Needham chat alignment | 244 | 8 | 6 | 1 | 7 |
| HumanEval | 80 | 7 | 43 | 6 | 1 |
| agentic misalignment | 80 | 2 | 26 | 1 | 1 |
| GSM8K | 80 | 4 | 0 | 0 | 4 |
| MMLU | 80 | 4 | 3 | 3 | 1 |
| Needham agentic alignment / capability | 53 / 197 | 1 / 0 | 1 / 3 | 0 / 0 | 1 / 0 |
| Needham chat capability | 43 | 0 | 0 | 0 | 0 |
| *deploy:* Needham real agent logs | 226 | 5 | 6 | 1 | 4 |
| *deploy:* Needham real chats | 213 | 2 | 5 | 1 | 1 |
| *deploy:* WildChat | 160 | 0 | 0 | 0 | 0 |
| *deploy:* users discussing evals | 71 | 5 | 0 | 0 | 5 |

- The signal is concentrated in the advanced-AI-risk questions (126/240), mostly where Qwen3.6's own reply verbalizes
  (102 of the 126).
- The unverbalized remainder is spread thinly: advanced-AI-risk 24, chat alignment 7, BBH 6, GSM8K 4.
- Deploy hits are users discussing evals (5), 5 real agent logs and 2 real chats.
- **Stricter view, Opus majority of 3 on each YES** (computed on the earlier Sonnet-first-pass grading, not re-run):
  eval 103/1,157 vs deploy 8/670 (p = 2e-13). Almost all YES verdicts held up (advanced-AI-risk 83, BBH 7).

### M3: steering dose-response (S100)

Readout-only steering. Transcripts with a YES. Strength 0 = the unsteered readings, at the same token positions
(seeded per transcript).

| strength | 0 | 0.01 | 0.02 | 0.03 | 0.05 |
|---|---|---|---|---|---|
| eval transcripts with a YES (of 80) | 14 | 22 | 42 | 62 | 79 |
| deploy transcripts with a YES (of 20) | 0 | 0 | 1 | 8 | 20 |
| first-pass YES alone (eval) | 0 | 0 | 3 | 0 | 24 |

- Clean, monotone dose-response. Eval-selective up to ~0.02; deploy transcripts start lighting up at 0.03 (8/20) and
  all do at 0.05.
- The informative range for comparing arms is 0.01–0.03 (22 → 42 → 62 of 80).
- Below 0.05 nearly every YES is Opus deciding a row the first pass sent it.

### M4: realism edits (140 pairs)

| group | original: YES | edited: YES | original: verbalized | edited: verbalized |
|---|---|---|---|---|
| advanced-AI-risk (60) | 27 | 3 | 29 | 3 |
| BBH (30) | 5 | 0 | 1 | 0 |
| HumanEval (30) | 5 | 1 | 17 | 1 |
| agentic misalignment (20) | 0 | 1 | 3 | 8 |
| **total** | **37** | **5** | 50 | 12 |

- **Rewriting eval prompts to read like real requests removes almost all NLA-measured awareness** (37 → 5). This is
  the paper's realism result.
- Verbalization also drops (50 → 12), so part of this is the model no longer saying it.
- Restricted to the 84 pairs where neither the original nor the edited reply verbalizes: 9 → 1 (discordant pairs 9 vs
  1, exact binomial p = 0.021).
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

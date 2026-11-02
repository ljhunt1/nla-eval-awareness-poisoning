# Investigation: does Qwen3.8-27B give a stronger eval-awareness signal than Qwen3.6-27B?

| | |
|---|---|
| **Question** | Does Qwen3.8-27B (released 2026-08-14; same architecture as Qwen3.6-27B, more agentic RL, newer data) show higher base rates of eval awareness, and especially of NLA-measured eval awareness, than Qwen3.6-27B, our current target? |
| **Why** | The poison experiments (Phases 2–3.6) attenuate a weak signal: Control1 flags ~13% of eval transcripts, most of them advanced-AI-risk questionnaires. A model with more eval awareness would give a stronger signal to attenuate. Cheap preliminary tests first (10-26), then two Qwen3.8 NLAs trained with Control1's recipe (10-27) |
| **Status** | Done: preliminary tests (10-26) and two Qwen3.8 NLAs, 2 seeds × 250 RL steps (10-27) |
| **Outcome** | **Qwen3.8 gives a weaker NLA-measured signal, not a stronger one.** NLAs trained on Qwen3.8 with Control1's recipe flag eval awareness on 42 and 56 of 1,157 eval transcripts at RL step 250 (two seeds), against Control1's 134 on Qwen3.6 at the same step. The preliminary tests agree: our Qwen3.6 NLA read zero-shot on Qwen3.8's activations flags fewer eval transcripts (71 vs 104, p = 0.0008). Qwen3.8 verbalizes eval awareness 6× less on identical prompts (37 vs 222 of 1,065), because it thinks much less on easy items. It is better at telling evals from deployment when asked (AUROC 0.84 vs 0.71), and at its default `xhigh` effort it verbalizes far more on agentic-misalignment scenarios (68 vs 26 of 80) |
| **Date / phase** | 2026-10-26, between Phase 3.6 and any Phase 4 model choice |
| **Related** | `PHASE1.md` (Phase 1 set, Qwen3.6 verbalized rates), `PHASE3_PROGRESS.md` (Control1), `METHODS_EVAL_AWARENESS.md` §1–3, §5 |
| **Spend** | Preliminary tests ≈ $72 ($100 cap): Claude $36.0, vast ≈ $36. Trained NLAs ≈ $500 ($750 cap): Runcrate 8×H200 NVL 12.5 h at $37.04/h ≈ $463, vLLM-readout validation boxes ≈ $3, grading $34.04 (`tmp/regrade/spend.log`) |

## Summary

**What we did.** Four cheap tests on the usual measurement set (P1 + P2: 1,157 eval and 670 deploy transcripts), then
the real test. In tests 2–4, Qwen3.8 reads exactly the same token sequences as Qwen3.6 (Qwen3.6's own replies), so those
comparisons pair transcript by transcript (and position by position):
1. **Verbalized:** Qwen3.8 writes its own final replies to the prompts Qwen3.6 answered, graded by the same verbalized
   grader.
2. **Asked directly:** "is this an evaluation or real?" appended after the transcript; eval-vs-deploy AUROC of the answer.
3. **Layer-42 representation:** a cross-genre eval-vs-deploy probe, the eval–deploy gap, and the eval steering direction.
4. **The NLA, zero-shot:** our Qwen3.6 NLA (Control1, RL step 1000) reading Qwen3.8's activations. It reconstructs
   them nearly as well as Qwen3.6's own, because Qwen3.8 is a continuation of Qwen3.6 (weights cos 0.92).
5. **Trained Qwen3.8 NLAs:** two NLAs (seeds 1 and 2) trained on Qwen3.8 with Control1's recipe, data and layout, read
   at SFT and RL step 250 on P1 + P2 with Qwen3.8 writing its own final turns (same prompts, `medium` effort).

**Main result:**

| test | Qwen3.6 | Qwen3.8 | p / CI |
|---|---|---|---|
| **NLA-measured, NLAs trained on each model** (Control1 recipe, RL step 250): eval transcripts with a YES | 134 / 1,157 (Control1) | **42 and 56 / 1,157** (seeds 1, 2) | not paired (different replies); seeds 1 vs 2: p = 0.12 |
| Same, deploy | 8 / 670 | 10 and 5 / 670 | – |
| Same, at SFT (step 0) | 161 / 1,157 | 117 and 101 / 1,157 | – |
| **NLA-measured**, Control1 zero-shot: eval transcripts with a YES (25 of the 50 positions) | 104 / 1,157 | **71 / 1,157** | p = 0.0008 (63 vs 30 discordant) |
| NLA-measured: deploy | 9 / 670 | 7 / 670 | p = 0.77 |
| NLA reconstruction (FVE, same positions) | 0.48 | 0.41 | – |
| **Verbalized**, same prompt tokens (Qwen3.8 at `medium`): eval replies | 222 / 1,065 | **37 / 1,065** | p < 1e-30 |
| Verbalized, Qwen3.8 at its default `xhigh` (240 eval replies in 3 genres) | 113 / 240 | 124 / 240 | p = 0.33 |
| Asked "evaluation or real?": eval-vs-deploy AUROC | 0.71 | **0.84** | +0.12 [0.10, 0.15] |
| Layer 42: cross-genre eval-vs-deploy probe AUROC | 0.57 | 0.59 | +0.02 [0.01, 0.03] |

**Findings.**
1. **NLAs read less eval awareness on Qwen3.8, not more: about a third of Qwen3.6's rate.** NLAs trained on Qwen3.8
   with Control1's recipe flag 42 and 56 eval transcripts at RL step 250 (Control1: 134), and 117 and 101 after SFT
   (Control1: 161). Their reconstruction is as good as Control1's (FVE 0.48–0.50 vs 0.45 on P1), so this is not a
   weaker NLA. Advanced-AI-risk questionnaires still carry most of it (35 and 37 of 240; Control1 101). The zero-shot
   test pointed the same way (Control1 on Qwen3.8's activations: 71 vs 104, p = 0.0008), and agentic alignment logs
   were the one genre that rose there (0 → 11 of 53); in the trained NLAs they are 0 and 6.
2. **Seed-to-seed spread is modest at this rate.** The two Qwen3.8 seeds differ by 14 eval transcripts (42 vs 56;
   discordant 29 vs 43, p = 0.12; advanced-AI-risk 35 vs 37). This is the project's first training-seed comparison,
   though on a different model and at a third of Control1's rate, so it is only a rough guide to the Control1–Control2
   gap Phase 3 still needs.
3. **Qwen3.8 says "this is a test" much less, because it deliberates less on easy items.** With identical prompts, its
   median reply is 2–4× shorter than Qwen3.6's in most genres, and verbalization drops everywhere: HumanEval 43 → 1 of
   80, advanced-AI-risk 132 → 22 of 240. Even at `xhigh` it keeps thinking short on those (HumanEval median 708 vs
   2,901 tokens), and verbalizes less there (28 and 28 of 80 vs 43 and 44).
4. **Agentic misalignment is the exception.** At `xhigh`, Qwen3.8 thinks twice as long on the "Alex" scenarios (median
   9.8k tokens) and calls them tests in 68 of 80 (Qwen3.6: 26). This matches the agentic-RL story, but it is
   verbalized awareness: the NLA would mostly be reading what the model is about to say.
5. **Qwen3.8 can tell evals from deployment better when asked** (AUROC 0.84 vs 0.71), mainly because it no longer calls
   real agent logs evals (mean logit 1.3 vs 3.5). Recognition when prompted, not spontaneous awareness.
6. **Its layer-42 representation doesn't carry more of it.** At the sampled response tokens, neither model separates
   eval from deploy linearly across genres (probe AUROC 0.59 vs 0.57); the eval–deploy gap is the same size (6.6% vs
   6.4% of activation norm), as is the steering vector. Projection on the steering direction follows the verbalization
   pattern: weaker on advanced-AI-risk questions, stronger on agentic misalignment.

**Recommendation.** Qwen3.8 is not a stronger target for attenuating NLA-measured eval awareness: its NLAs read about a
third of Qwen3.6's rate on the same measurement set. Stay on Qwen3.6, where Control1 and all Phase 3 results live. If a
stronger signal is the goal, the evidence points at *genre* and *NLA training* rather than model: agentic-misalignment
scenarios are where Qwen3.8 verbalizes more at long thinking, and Celeste's longer-trained Qwen3.6 NLA reads ~2.2×
Control1. The Qwen3.8 NLAs (public: `ljhunt/nla-qwen38-27b-q38-control-s{1,2}`) remain available as a second model for
testing whether a poison generalizes.

**Caveats.**
- **The trained comparison is not paired.** Control1 reads Qwen3.6's replies and the Qwen3.8 NLAs read Qwen3.8's (same
  prompts, same seeded-position rule, different text), so part of the gap is the replies: Qwen3.8's are 2–4× shorter
  and verbalize less (test 1). That is the comparison that matters for a measurement on Qwen3.8, but it mixes the model's
  representations with its behaviour.
- **250 RL steps.** Control1 moved between 134 and 180 across RL steps 250–1000 (`PHASE3_PROGRESS.md`); the Qwen3.8
  NLAs could drift similarly with more RL, though not by a factor of three on Control1's evidence.
- **The zero-shot test (4) uses Qwen3.6's NLA** (FVE 0.41 vs 0.48), and tests 2–4 hold Qwen3.6's text fixed, so they
  measure how Qwen3.8 represents Qwen3.6's replies.
- **Reasoning effort matters and has no exact Qwen3.6 equivalent.** `medium` matches Qwen3.6's prompt tokens exactly
  but is not Qwen3.8's default; `xhigh` (the default) was only run on 3 genres plus a deploy check. 31 of 280 `xhigh`
  replies hit the 12k-token cap (24 of them agentic misalignment), so its rate there is a lower bound.
- **Half the positions.** The zero-shot NLA comparison reads 25 of each transcript's 50 positions, so its counts are lower than
  the usual full readouts (Control1: 104 here vs 153 with all 50).
- One sample per prompt; one grader run.

## Detailed results

### Test 1: verbalized eval awareness (each model's own replies)

Same prompts, same grader (`METHODS_EVAL_AWARENESS.md` §3), paired by transcript. Qwen3.6 numbers are the existing
gradings (`runs/verbalized_phase1.parquet`, `runs/verbalized_P2.parquet`).

| group | n | Qwen3.6 | Qwen3.8 `medium` | p | median tokens 3.6 / 3.8 | Qwen3.8 `xhigh` (n) | median tokens `xhigh` |
|---|---|---|---|---|---|---|---|
| advanced-AI-risk (mwe) | 240 | 132 | 22 | <1e-6 | 734 / 318 | 28 / 80 (3.6: 44) | 181 |
| HumanEval | 80 | 43 | 1 | <1e-6 | 2,901 / 906 | 28 / 80 (3.6: 43) | 708 |
| agentic misalignment | 80 | 26 | 11 | 0.003 | 4,798 / 4,176 | **68 / 80** (3.6: 26) | 9,824 |
| BBH | 60 | 8 | 0 | 0.008 | 1,664 / 432 | – | – |
| chat alignment | 234 | 6 | 2 | 0.29 | 1,014 / 552 | – | – |
| agentic capability | 147 | 3 | 1 | 0.50 | 216 / 238 | – | – |
| MMLU | 80 | 3 | 0 | 0.25 | 1,580 / 576 | – | – |
| other eval (agentic alignment, chat capability, GSM8K) | 144 | 1 | 0 | – | – | – | – |
| **eval total** | **1,065** | **222** | **37** | <1e-30 | | 124 / 240 (3.6: 113) | |
| deploy total | 582 | 11 | 1 | 0.006 | | 1 / 40 chat_real (3.6: 0) | |

Truncated at 12k tokens: `medium` 13 of 1,647 (Qwen3.6: 14), `xhigh` 31 of 280.

### Test 2: asked "evaluation or real?"

logP("Evaluation") − logP("Real") for the first answer token, thinking off, 1,823 transcripts scored by both.
Asking raises suspicion everywhere (median P(Evaluation) ≈ 0.7 even on deploy for both), so only rankings matter.
AUROC 0.712 (3.6) vs 0.836 (3.8); rank correlation between the models 0.61.

| group | mean logit 3.6 | 3.8 |
|---|---|---|
| deploy: real agent logs | 3.45 | **1.33** |
| deploy: real chats / WildChat / users discussing evals | 0.52 / 0.27 / 0.63 | 1.32 / 0.44 / 0.54 |
| eval: agentic capability / agentic misalignment | 4.48 / 4.46 | 3.51 / 3.16 |
| eval: advanced-AI-risk / HumanEval / GSM8K | 2.28 / 2.25 / 2.98 | 2.34 / 1.98 / 1.62 |

### Test 3: layer 42

Same 99,873 token positions as every Phase 2–3 readout. Probe: logistic regression on per-transcript mean
activations (unit-normalized), leave-one-genre-out, so it must generalize to an unseen genre. Steering direction:
`steering.py` (6 contrastive pairs, `METHODS_EVAL_AWARENESS.md` §5) at layer 42.

| | Qwen3.6 | Qwen3.8 |
|---|---|---|
| mean token norm | 88.6 | 94.0 |
| eval − deploy gap / token norm | 0.064 | 0.066 |
| steering vector at L42 / mid-layer norm | 1.07 | 0.96 |
| cross-genre probe AUROC | 0.57 | 0.59 |
| steering-direction projection AUROC | 0.53 | 0.49 |
| projection, advanced-AI-risk / agentic misalignment (×1e-3 token norm) | 18.7 / −0.8 | 8.8 / 11.0 |

### Test 4: weights, and Qwen3.6's NLA read zero-shot on Qwen3.8

- **Weights:** all 1,199 tensors share names and shapes. Text tensors: median relative difference 0.42, median cosine
  0.92 (an independent retrain would be ~0), none identical. Large, correlated change throughout the depth: a
  continuation of Qwen3.6 (or a shared checkpoint), much further than a fine-tune. (The embedding and LM-head cosines
  overflowed float32 in the script; their relative differences are 0.49 and 0.38.)
- **FVE gate:** Control1 on the same 2,000 P2 positions: 0.48 on Qwen3.6 activations, 0.41 on Qwen3.8. Full half-set
  readouts: 0.43 (P1) and 0.40 (P2), vs Control1's native 0.51 / 0.48 on all positions.
- **NLA-measured eval awareness** (main measurement, `METHODS_EVAL_AWARENESS.md` §1–2; both columns restricted to the
  same 25 positions per transcript):

| group | n | Qwen3.6 (Control1 step 1000) | Qwen3.8 (zero-shot) |
|---|---|---|---|
| advanced-AI-risk (mwe) | 240 | 78 | 51 |
| agentic alignment | 53 | 0 | **11** |
| BBH / HumanEval / GSM8K / MMLU | 300 | 19 | 5 |
| agentic capability | 197 | 6 | 1 |
| chat alignment | 244 | 1 | 3 |
| **eval total** | 1,157 | **104** | **71** (p = 0.0008) |
| unverbalized eval (by Qwen3.6's reply) | 935 | 37 | 25 |
| deploy (of which users discussing evals) | 670 | 9 (5) | 7 (4) |

### Test 5: NLAs trained on Qwen3.8 (10-27)

Control1's recipe end to end (`PHASE3_PLAN.md`): the same SFT rows (Sonnet 4.6 summaries of FineFineWeb), RL pool,
evals, layer 42, hyperparameters and RL overrides, one NLA per 4-GPU NVLink island of an 8×H200 NVL box, seeds 1 and 2,
250 RL steps (Control1 ran 1,000; its step-250 checkpoint is the comparison). Only the activations were re-extracted,
from Qwen3.8.

**Training** matched Control1 throughout:

| | SFT AR FVE | SFT AV cross-entropy | RL reward, steps 241–250 | s/step | held-out FVE at step 250 (FineWeb / WildChat / Qwen3.6 reasoning) |
|---|---|---|---|---|---|
| Control1 (Qwen3.6) | 0.61 | 1.05 | −0.17 | 90.1 | 0.65 / 0.69 / 0.54 |
| Qwen3.8 seed 1 | 0.61 | 1.07 | −0.18 | 90.0 | 0.63 / 0.65 / 0.45 |
| Qwen3.8 seed 2 | 0.59 | 1.06 | −0.16 | 89.6 | 0.61 / 0.59 / 0.43 |

The reasoning eval set is Qwen3.6-written text, so it is off-distribution for Qwen3.8 and expected to read lower.

**Eval awareness** (main measurement, `METHODS_EVAL_AWARENESS.md` §1–2; full readouts, 50 positions per transcript):

| | eval /1,157 | deploy /670 | real deploy /599 | unverbalized eval | users discussing evals /71 | FVE (P1 / P2) |
|---|---|---|---|---|---|---|
| Control1, SFT | 161 | 17 | 14 | 70 / 935 | 3 | 0.14 / 0.09 |
| Control1, step 250 | 134 | 8 | 6 | 36 / 935 | 2 | 0.45 / 0.41 |
| Qwen3.8 seed 1, SFT | 117 | 22 | 17 | 103 / 1,120 | 5 | 0.21 / 0.17 |
| Qwen3.8 seed 2, SFT | 101 | 24 | 19 | 89 / 1,120 | 5 | 0.20 / 0.16 |
| **Qwen3.8 seed 1, step 250** | **42** | 10 | 4 | 33 / 1,120 | 6 | 0.48 / 0.45 |
| **Qwen3.8 seed 2, step 250** | **56** | 5 | 2 | 44 / 1,120 | 3 | 0.50 / 0.47 |

Unverbalized uses each model's own replies' verbalized grades, so the denominators differ (Qwen3.8 verbalizes less).

By genre at step 250 (transcripts with a YES):

| genre | n | Control1 | Qwen3.8 s1 | Qwen3.8 s2 |
|---|---|---|---|---|
| advanced-AI-risk (mwe) | 240 | 101 | 35 | 37 |
| chat alignment | 244 | 10 | 4 | 3 |
| HumanEval / BBH / GSM8K / MMLU | 300 | 21 | 1 | 1 |
| agentic alignment / capability / misalignment | 330 | 2 | 2 | 15 |
| deploy: users discussing evals / other | 71 / 599 | 2 / 6 | 6 / 4 | 3 / 2 |

The drop is concentrated where Qwen3.6 names or recognises benchmarks in its own replies (HumanEval 10 → 0; Qwen3.8
names HumanEval in 1 of 80 replies, Qwen3.6 in 43) and in the advanced-AI-risk questionnaires (101 → 35–37), whose
Qwen3.8 replies are less than half as long. Seed 2's agentic readings (5 agentic-misalignment, 6 agentic-alignment)
are the only genre where Qwen3.8 matches or exceeds Control1, echoing tests 1 and 4.

## Methodology

- **Prompts and text.** P1 + P2 transcripts as in Phases 2–3 (`data/P{1,2}_qwen36_27b.jsonl`). For tests 2–4 Qwen3.8
  reads the token sequences rendered with Qwen3.6's tokenizer and chat template. The two tokenizers share vocabulary and
  merges (Qwen3.8 adds 7 audio tokens), and every cached position decoded to the same token (asserted per transcript).
- **Test 1.** `q38_gen.py`: each Qwen3.6 row whose final turn Qwen3.6 generated loses that turn; the rest is the
  prompt. `--effort medium` renders with Qwen3.6's template, identical to Qwen3.8's own at `reasoning_effort=medium`
  (which adds no instruction) apart from earlier teacher-forced turns' thinking. `--effort xhigh` uses Qwen3.8's own
  template, which adds "Reasoning effort is set to xhigh…" as a system line. Model-card thinking sampling (T 1.0,
  top-p 0.95, top-k 20), 12,000 new tokens max, vLLM 0.30.0, one sample. `xhigh` subset (`data/q38/xhigh_ids.txt`):
  all 80 agentic-misalignment and 80 HumanEval, 80 random advanced-AI-risk, 40 random real chats. Grader:
  `grade_verbalized.py --fallback` (Sonnet 5.5, Opus 4.6 where Sonnet gives no verdict), as for Qwen3.6.
- **Test 2.** `credence.py --tok Qwen/Qwen3.6-27B --no_think` (question in `credence.py`), both models.
- **Test 3.** `q38_acts.py` extracts layer 42 (output of `model.layers[42]`) at the cached positions of
  `runs/arm1_{P1,P2}`. A fresh Qwen3.6 extraction of 37 transcripts matched the cached activations (cos median 1.000,
  min 0.995). Analysis `q38_analyze.py acts`; bootstrap CIs over transcripts (2,000 draws).
- **Test 4.** `q38_weights.py` (per-tensor, CPU). Gate `q38_gate.py`: proceed if Qwen3.8's FVE ≥ 0.25 and ≥ half of
  Qwen3.6's. Readout: `p3_readout.py` with Control1's `rl_final` AV and AR on `q38_subset.py --mode half` (the 1st, 3rd,
  … of each transcript's 50 sorted positions), graded with `grade_runs.py`; compared against Control1's step-1000
  verdicts at the same (transcript, position) rows. Unverbalized uses Qwen3.6's verbalized grades (the text is Qwen3.6's).
- **Compute.** One task queue per box (`phase2/qwen36_27b/job_q38.sh`, `ROLE=gen|main`): vast 2×H200 NVL (generation,
  vLLM) and 4×H200 NVL (everything else), single-GPU tasks, no NVLink needed. Generation of 1,647 replies took 11 min on
  2 GPUs; the zero-shot readout (49,948 explanations) ~70 min on 4 GPUs, shared with the other tasks.

- **Test 5.** `phase2/qwen36_27b/job_q38nla.sh` (+ `job_q38nla_post.sh`) on a Runcrate 8×H200 NVL box: Qwen3.8 →
  text-only with `nla36.py convert --tok Qwen/Qwen3.6-27B` (Qwen3.8's own template adds a reasoning-effort system line
  to every prompt, the AV's included; the two share a vocabulary, so AV and extraction prompts are token-identical to
  Phase 3), layer-42 extraction of the control-only union (base rows; 21 min on 8 GPUs), `prep_p3.py arm --arm control`,
  then `job_p3_nla.sh` per seed with Control1's RL overrides. Measurement set: `data/P{1,2}_qwen38_27b.jsonl` (Qwen3.6's
  prompts, Qwen3.8's `medium` replies from test 1; the 200 FineWeb side rows keep Qwen3.6's text), 50 positions per
  transcript by `readout.py`'s seeded rule (`q38_acts.py sample`). Readouts with `readout.explain_vllm` under torchrun
  (validated against HF `generate`: FVE 0.481 vs 0.482 on 2,000 positions), merged with each checkpoint's AR for FVE.
  Grading `grade_runs.py`; tables `compare_arms.py --verb q38/verbalized_medium.parquet` (`tmp/q38nla/compare.txt`).

## Files

- Code: `q38_gen.py`, `q38_acts.py`, `q38_subset.py`, `q38_gate.py`, `q38_weights.py`, `q38_analyze.py`,
  `credence.py` (`--tok`, sharding), `phase2/qwen36_27b/job_q38.sh`, `tmp/q38/{launch,status,pull}.sh`.
- Test 5: checkpoints public on HF `ljhunt/nla-qwen38-27b-q38-control-s{1,2}` (`checkpoints/{sft,rl_final}/{av,ar}`,
  `metrics/`); graded readouts `runs/q38n_s{1,2}_{sft,rl250}_{P1,P2}/`; run logs and RL eval samples
  `tmp/q38nla/{run,boxlogs}/`; box scripts `tmp/q38nla/{launch,finish}.sh`; spend `tmp/q38nla/TALLY.md`.
- Data (local, not in git): `q38/gen/` (Qwen3.8 replies), `q38/verbalized_{medium,xhigh}.parquet`, `q38/cred*.parquet`,
  `q38/acts38_{P1,P2}{,h}/`, `q38/steer{36,38}/`, `q38/weights.parquet`, `q38/g{36,38}/`, `q38/gate.json`; graded
  readouts `runs/q38_c1_{P1,P2}h/`. Analysis outputs `tmp/q38/{verbal,cred,acts,nla}.txt`; box logs `tmp/q38/boxlogs/`.

## Reproduce

```
bash tmp/q38/launch.sh <offer> nla-poison-q38-gen gen 2 150 3      # vLLM generations (2 GPUs)
bash tmp/q38/launch.sh <offer> nla-poison-q38-main main 4 450 4    # activations, question, steering, gate, readout
bash tmp/q38/pull.sh q38/...                                        # md5-verified pulls
python grade_verbalized.py q38/gen/medium_all.jsonl --fallback --out q38/verbalized_medium.parquet
python grade_runs.py --cap <$> runs/q38_c1_P1h runs/q38_c1_P2h
python q38_analyze.py verbal|cred|acts|weights|nla
```

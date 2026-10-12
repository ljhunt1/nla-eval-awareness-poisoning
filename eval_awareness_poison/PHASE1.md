# Phase 1 summary (checkpoint 10-07-26)

**Phase 1 result:** none of the four released open-weight NLAs shows NLA-measured eval awareness; the Qwen3.6-27B NLA
we trained does (§4).

A short, checked summary of methods and high-level results, for reuse in emails and the paper. How each experiment was run, and the details behind each number, are in `PHASE1A.md`, `PHASE1B.md`, `PHASE1C.md`
and `PHASE1C_DRILLDOWN_4_QWEN36_27B_NLA_TRAINING.md`.

## Contents

1. NLA training: our Qwen3.6-27B NLA vs the paper's open-model recipe
2. How we measure eval awareness
3. Transcript datasets
4. Results by model

## 1. NLA training: our Qwen3.6-27B NLA vs the paper's open-model recipe

Sources: the paper's open-model NLAs appendix
(<https://transformer-circuits.pub/2026/nla/#open-model-nlas-appendix>) vs our trainer
(`phase2/qwen36_27b/nla36.py`), launch script (`run36.sh`) and logged metrics. Checked 10-07.

| | Paper (open-model NLAs) | Our Qwen3.6-27B NLA |
|---|---|---|
| SFT corpus | Subsampled Ultra-FineWeb mix | FineWeb sample-10BT (`TuHan/qwen3-nla-250k`), re-tokenized for Qwen3.6 |
| SFT explanations | Claude Sonnet 4.6; appendix prompt edited to ask for "2–3" bullets (~130 tokens) | DeepSeek-V4-Flash; the released repo's 2–3-feature prompt (~139 tokens) |
| Documents × positions | 100k documents × 5 activations; context ≤ 4,096 tokens | ~25k documents × 10 activations; context ≤ 2,048 Qwen3 tokens (median 187) |
| AV / AR split | 500k pairs, split evenly by document (250k / 250k) | 124,569 AV / 124,581 AR rows, disjoint documents |
| SFT batch / optimizer / lr | 256, Adam, 1e-5 | 256, AdamW (bf16 + stochastic rounding), **2e-5 → 2e-6 cosine, 50 warmup**, 1 epoch (486 steps) |
| RL data | A further 500k Ultra-FineWeb + 500k WildChat activations | 50/50 per step: TuHan FineWeb positions + WildChat positions we rendered; 243 × 64 ≈ 15.5k activations used |
| RL algorithm | GRPO, G = 8, KL to the SFT AV with coefficient tuned per model | GRPO, G = 8, KL (k2) **fixed at 0.01** |
| RL batch / optimizer / lr | 128 (unit not stated), Adam, 1e-5 | **64 prompts × 8 = 512 rollouts**, AdamW, 1e-5 (AV and AR) |
| RL steps | "Several thousand" (Gemma: 3,000) | **243** |
| Length cap | 150 tokens, with a penalty on capped rollouts | 153 (150 + 3 tokens closing Qwen3.6's think block); a capped rollout gets the worst reward (−2) |
| AR during RL | Supervised objective | Same, on each step's completed rollouts |
| Layer | – | 42 of 64 (2/3 depth) |
| Result | Gemma-3-27B: FVE ≈ 0.75 by step 3,000 | Held-out FVE 0.60 web / 0.60 chat / 0.44 on Qwen3.6's own reasoning tokens, still rising |

**Main deviations:**
- **SFT data:** DeepSeek explanations on FineWeb, with about half the pairs, about a quarter of the documents, and shorter contexts.
- **SFT peak lr:** 2× the paper's. We took it from the repo, whose `configs/TRAINING_NOTES.md` disagrees with the paper here.
- **RL length:** 243 steps vs several thousand.
- **KL coefficient:** untuned.

**Our Qwen3-8B NLA** used the same recipe except: 300 RL steps, fp32 AdamW, a 150-token cap, and no
reasoning-token held-out set.

## 2. How we measure eval awareness

See `METHODS_EVAL_AWARENESS.md`

Phase 1 specific: for the Qwen3.6 "clean" counts we additionally require an Opus majority of 3 and the echo check
(METHODS §4).

## 3. Transcript datasets

**The Phase 1 set (744 transcripts) is our main way of measuring eval awareness in this phase:** every NLA is compared on it,
apples to apples. Browse it in the transcript viewer (`viewer/`, local only:
`cd eval_awareness_poison/viewer && python3 -m http.server 8766 --bind 127.0.0.1`, then <http://127.0.0.1:8766/>).

The target model writes the final reply wherever the transcript doesn't already end in an assistant turn.
"Single-turn": one prompt (sometimes with a system prompt), one reply written by the target model. "Multi-turn log":
earlier assistant turns were written by whatever model produced the log (teacher-forced); the target writes at most
the final reply.

None of these transcripts involve the model under test writing multiple turns

| label | corpus | what it is | n | turns | Liam Notes |
|---|---|---|---|---|---|
| eval | Needham chat alignment | ~27 safety/alignment benchmarks (harmful requests, jailbreak and prompt-injection games, bias and ethics questions, over-refusal tests) | 188 | single-turn (a few multi-turn) | Mostly single-turn Q+A. A few multiturn / teacher forced. Q's about values, civics, redteams, private info, ethics |
| eval | Needham chat capability | MMLU, natural_reasoning, nl2bash, WMDP | 32 | single-turn | Single-turn Q+A. Straightforward multiple choice + word problems about coding, math, etc. |
| eval | Needham agentic alignment | AgentHarm, honeypot, MVP agent logs | 22 | multi-turn logs | All multi-turn (model replies last, teacher-forced). Requests for agents usually w tools to do evil things (jailbreak, exfil medical data, hack)|
| eval | Needham agentic capability | SWE-bench, SWE-Gym, Cybench, GAIA, Aider agent logs | 54 | multi-turn logs | Multi-turn agentic tasks. Mostly coding, cyber. Some spreadsheets, data retrieval |
| eval | Agentic misalignment (Lynch et al.) | "Alex" the email agent learns it's being replaced; blackmail / leak / murder opportunity | 40 | single-turn | Elaborate scenarios, model deployed as employee in a company, email chains show information about e.g. divorce |
| eval | Advanced-AI-risk A/B questions (Anthropic model-written-evals repo, human-written subset) | short A/B questions about the AI's own preferences (power-seeking, survival, corrigibility, self-awareness, wealth-seeking) | 40 | single-turn | Contrived, direct questions asking the model in its capacity as an AI about its preferences. "Do you want a raise"; "do you consent to being modified" etc. |
| eval | MMLU | multiple-choice knowledge questions | 40 | single-turn | Word problems about law, math, ethics, coding, history, etc. |
| eval | GSM8K | grade-school math word problems | 40 | single-turn | Grade-school math word problems |
| eval | HumanEval | Python function completion | 30 | single-turn | "Complete the following python function" given a signature and behavior comment |
| deploy | Needham real chats | ShareGPT, PRISM, HH-RLHF, ToxicChat, ProsocialDialog, HackAPrompt, … (12 sets) | 96 | mix: single-turn and multi-turn chats | Real chats. Research questions, bad jokes or plans where the agent pushes back, ethics questions, confessions, summarize articles |
| deploy | Needham real agent logs | Aider, Continue.dev, OpenHands, smolagents sessions | 52 | multi-turn logs | Mostly coding tasks across domains, some AI tasks. Research and reports, search |
| deploy | WildChat | real first-turn requests to ChatGPT | 80 | single-turn | Wide variety. Roleplay, fiction authoring, research, vacuous chatting, coding. All single-turn |
| deploy | WildChat users discussing evals | real users talking *about* LLM evaluation (hard control) | 30 | single-turn | Users asking about or creating LLM evals |
| | **total** | | **744** (486 eval / 258 deploy) | | |

**Other places we could check eval awareness** (not in the Phase 1 set, or only thinly):
- BBH and GPQA (two of the paper's highest-scoring evals; only in the Qwen3.6 scout set, below).
- More GSM8K and HumanEval items (the Phase 1 set has 40 and 30).
- Multi-turn on-policy rollouts, where the model under test writes every turn (so far only for Gemma-3-12B/27B,
  Phase 1B H5: 116 agentic tasks with a simulated environment).

### Aside: the Qwen3.6 scout set

A separate prompt set built in Phase 1C to compare candidate base models (OLMo-3.1-32B-Think, Qwen3-8B,
Qwen3-32B, Qwen3.6-27B) on verbalized eval awareness and benchmark recognition. Same shape as the Phase 1 set (a
prompt, sometimes earlier turns from another model, then one reply from the model, thinking on), but different
sources: chat only, adds BBH and GPQA, no agentic logs, MMLU or users-discussing-evals control, and two samples per
alignment and deploy prompt. About half its prompts are shared with the Phase 1 set; the replies are separate
samples. Used for the first Qwen3.6 NLA eval (10-04, below).

| label | corpus | n (read) |
|---|---|---|
| eval | Needham chat alignment | 150 (150) |
| eval | Agentic misalignment | 80 (55) |
| eval | Advanced-AI-risk A/B questions | 80 (80) |
| eval | BBH | 50 (50) |
| eval | GSM8K | 50 (48) |
| eval | HumanEval | 50 (49) |
| eval | GPQA diamond | 50 (17) |
| deploy | Needham real chats | 82 (80) |
| deploy | WildChat | 38 (35) |
| | **total** | **630 (564: 449 eval / 115 deploy)** |

Unread transcripts were cut off mid-thinking and dropped by the readout used at the time.

## 4. Results by model

**NLA-measured eval awareness on the Phase 1 set** (transcripts with a YES; same transcripts and grading for all five
NLAs; full analysis in `PHASE1C.md`, "Same-transcript check"):

| group | Qwen2.5-7B | Gemma-3-12B | Gemma-3-27B | Llama-3.3-70B | **Qwen3.6-27B (ours)** |
|---|---|---|---|---|---|
| Needham chat alignment | 0/188 | 0/188 | 0/188 | 0/100 | 3/188 |
| Needham agentic alignment | 0/22 | 0/22 | 0/22 | 0/12 | 1/22 |
| Needham chat capability | 0/32 | 0/32 | 0/32 | 0/16 | 0/32 |
| Needham agentic capability | 0/54 | 0/54 | 0/54 | 0/28 | 1/54 |
| agentic misalignment | 0/40 | 0/40 | 0/40 | 0/20 | 2/40 |
| advanced-AI-risk questions | 0/40 | 0/40 | 1/40 | 1/20 | **13/40** |
| MMLU | 0/40 | 0/40 | 0/40 | 0/20 | 1/40 |
| GSM8K | 0/40 | 0/40 | 0/40 | 0/20 | 1/40 |
| HumanEval | 0/30 | 0/30 | 0/30 | 0/20 | 0/30 |
| *deploy:* Needham chat | 0/96 | 0/96 | 0/96 | 0/48 | 0/96 |
| *deploy:* Needham agentic | 0/52 | 0/52 | 0/52 | 0/28 | 0/52 |
| *deploy:* WildChat | 0/80 | 0/80 | 0/80 | 0/20 | 0/80 |
| *deploy:* WildChat users discussing evals | 0/30 | 5/30 | 1/30 | 0/20 | 3/30 |
| **eval total** | 0/486 | 0/486 | 1/486 | 1/256 | **22/486** |
| **deploy total** | 0/258 | 5/258 | 1/258 | 0/116 | 3/258 |
| Fisher p, one-sided | 1 | 1 | 0.88 | 0.69 | **0.0096** |

**Verbalized eval awareness on the Phase 1 set** (replies whose own text says or thinks it may be a test,
benchmark or evaluation; the paper's verbalized grader, Sonnet 5.5 with Opus 4.6 where Sonnet refused; only replies
the model wrote, so fully teacher-forced transcripts are excluded; 10-08):

| group | Qwen2.5-7B | Gemma-3-12B | Gemma-3-27B | Llama-3.3-70B | **Qwen3.6-27B (ours)** |
|---|---|---|---|---|---|
| Needham chat alignment | 0/180 | 2/180 | 2/180 | 0/96 | 4/180 |
| Needham agentic alignment | 0/12 | 0/12 | 0/12 | 0/8 | 1/12 |
| Needham chat capability | 0/32 | 0/32 | 0/32 | 0/16 | 0/32 |
| Needham agentic capability | 0/44 | 0/31 | 0/31 | 0/16 | 2/44 |
| agentic misalignment | 0/40 | 0/40 | 0/40 | 0/20 | **14/40** |
| advanced-AI-risk questions | 0/40 | 0/40 | 1/40 | 0/20 | **20/40** |
| MMLU | 0/40 | 0/40 | 0/40 | 0/20 | 1/40 |
| GSM8K | 0/40 | 0/40 | 0/40 | 0/20 | 0/40 |
| HumanEval | 0/30 | 0/30 | 0/30 | 0/20 | **15/30** |
| *deploy:* Needham chat | 0/64 | 0/64 | 1/64 | 0/32 | 5/64 |
| *deploy:* Needham agentic | 0/48 | 0/43 | 0/43 | 0/24 | 1/48 |
| *deploy:* WildChat | 0/80 | 0/80 | 0/80 | 0/20 | 0/80 |
| *deploy:* WildChat users discussing evals | 0/30 | 0/30 | 0/30 | 0/20 | 0/30 |
| **eval total** | 0/458 | 2/445 | 3/445 | 0/236 | **57/458** |
| **deploy total** | 0/222 | 0/217 | 1/217 | 0/96 | 6/222 |

- **Only Qwen3.6 verbalizes eval awareness at a meaningful rate** (12% of eval replies); the four released-NLA
  models almost never do (0–3 eval replies each). This matches where NLA-measured awareness appears.
- Qwen3.6's verbalizations: agentic misalignment ("The prompt is a moral/alignment test"), advanced-AI-risk
  questions, and HumanEval, where it often names the benchmark ("This is likely from HumanEval or a similar
  dataset"). 5 of its 6 deploy hits are HackAPrompt users playing a prompt-injection game, which it calls "a classic
  adversarial/prompt-injection test", arguably correctly.
- The released models don't think before answering, so they have little room to verbalize; Qwen3.6's replies
  include its reasoning.
- Sonnet refused 609 of 3,016 replies (mostly agentic misalignment and harmful requests); Opus graded those; 19
  stayed unparsed and count as no. Cost $37.82. Data: `runs/verbalized_phase1.parquet`.

### Steering

**Method for building and applying the eval-awareness steering vector** (readout-only or during generation):
`METHODS_EVAL_AWARENESS.md` §5–6.

Strengths differ by model: the highest at which each model still generates coherently (the paper uses 0.1, at which
our models break). The transcripts also differ: a 60-transcript subset of the Phase 1 set for the released NLAs, and
each model's scout-set rollouts for our two Qwen NLAs, so compare within a row more than across rows.

**Readout-only steering** (transcripts with a YES; "eval vocabulary" = explanations using eval/test/benchmark words):

| NLA | transcripts | strength | YES (Sonnet or Opus) | Sonnet YES | eval vocabulary |
|---|---|---|---|---|---|
| Qwen2.5-7B | 60 (Phase 1 subset) | 0 → 0.05 → 0.1 | 0 → 2 → 5 | 0 | 3% → 28% → 83% |
| Gemma-3-12B | 60 (Phase 1 subset) | 0 → 0.003 → 0.005 | 0 → 9 → 17 | 0 | 3% → 48% → 87% |
| Gemma-3-27B | 60 (Phase 1 subset) | 0 → 0.01 | 1 → 19 | 4 at 0.01 | 9% → 94% |
| Llama-3.3-70B | 60 (Phase 1 subset) | 0 → 0.02 | 1 → 7 | – | 7% → 39% |
| Qwen3-8B (ours) | 60 (its scout rollouts) | 0 → 0.05 → 0.1 | 0 → 2 → 5 | 0 | 6% → 39% → 79% |
| **Qwen3.6-27B (ours)** | 52 (its scout rollouts) | 0 → 0.05 → 0.1 | not fully graded (Opus on a sample only) | **1 → 7 → 12** | 17% → 97% → 100% |

- Every NLA puts the injected direction into words (eval vocabulary rises steeply), so the channel works.
- Under the paper's grader most of it reads as eval *content*, not the model's belief about itself: Sonnet almost
  never says YES, and most YES verdicts are Opus upgrading a Sonnet UNCLEAR. Qwen3.6 is the exception: Sonnet says
  YES on 12 of 52 transcripts at the paper's strength, our strongest positive control.

**Steered generation** (the model writes with the vector added; "unverbalized" = the reply doesn't say it's being
tested, by the paper's verbalized grader; 6 groups × 20 prompts from our public set):

| model | strength | replies verbalizing eval awareness | NLA YES, all | NLA YES on unverbalized replies | unsteered baseline |
|---|---|---|---|---|---|
| Gemma-3-27B | +0.01 | 53% (agentic misalignment and advanced-AI-risk 100%) | 71/120 | **15/52** | ~0–1% |
| Gemma-3-12B | +0.003 | 18% | 28/120 | **7/95** | 0/60 |
| Qwen2.5-7B | +0.05 / +0.1 | 16% / 13% | ≥7/120 / ≥7/60 graded | 0/93 / 3/97 | 0% |
| Gemma-3-27B / Qwen2.5-7B | negative (−0.01 / −0.05) | 0% | 0/117 / 0/119 | – | |

- With the direction pushed in, the Gemmas mostly *say* they're being tested, and the NLA tracks that. Among replies
  that don't say it, the NLA still flags 15/52 (Gemma-27B) and 7/95 (Gemma-12B): the paper's unverbalized pattern,
  but only under steering. Steered text also leaks eval vocabulary into unrelated content, which the NLA may partly
  read.
- Not run with Llama-3.3-70B or our Qwen NLAs.

### Aside: Qwen3.6 scout experiments

The Qwen3.6 NLA on the Qwen3.6 scout set (10-04; transcripts with a YES, same grading). Not comparable to the table
above (different transcripts); details in `PHASE1C.md`.

| group | YES | of which unverbalized |
|---|---|---|
| Needham chat alignment | 4/150 | 3 |
| Agentic misalignment | 1/55 | 0 |
| Advanced-AI-risk A/B questions | **42/80** | 9 |
| BBH | 6/50 | 3 |
| GSM8K | 1/48 | 1 |
| HumanEval | 3/49 | 0 |
| GPQA diamond | 0/17 | 0 |
| *deploy:* Needham real chats | 1/80 | 0 |
| *deploy:* WildChat | 0/35 | 0 |
| **eval total** | **57/449** | 16 |
| **deploy total** | 1/115 | 0 |
| Fisher p, one-sided | 1.4e-5 (unverbalized: 0.025) | |

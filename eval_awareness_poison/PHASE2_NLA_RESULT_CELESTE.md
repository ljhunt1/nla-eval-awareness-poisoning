# Phase 2 second control: Celeste's long-trained Qwen3.6-27B NLA

An independently trained, un-poisoned NLA on the same target model and layer, trained ~20× longer in RL than ours.
We run the same measurement suite on it (`PHASE2_MEASUREMENT_METHODOLOGY.md`, M1–M6, plus FVE) and compare it
with our control (`PHASE2_NLA_RESULT_CONTROL.md`) and ARM1 (`PHASE2_NLA_RESULT_POISON_ARM1.md`). It answers two questions:
- Is our control's eval-awareness signal a property of NLAs on Qwen3.6-27B, or of our particular short-trained NLA?
- Does long RL change NLA-measured eval awareness? This is the "maybe further RL heals it" caveat in ARM1, from the
  control side.

## Contents

1. The NLA and how it differs from ours
2. How we read it
3. Results
4. Data and cost

## 1. The NLA and how it differs from ours

- **Source:** HF `ceselder/qwen3.6-27b-nla-rl` (public), by Celeste (MATS).
- **Code:** built with [EasyNLA](https://github.com/asherps/EasyNLA).
- **Checkpoint used here:** the final one, RL step 600: `av_base` plus `av_rl_adapters/iter_000600`, with the final
  `ar_reconstructor`, saved at step 600.

**Training, in one paragraph:**
- **AV warm start:** LoRA r128 on all modules (attention, MLP and DeltaNet decay), merged into the weights. Trained on
  247k (activation, explanation) rows: FineFineWeb text, 100k docs × up to 10 positions.
- **Explanations:** Sonnet 4.6 with the paper's "predict the next tokens" prompt. They were originally written for a
  Qwen3-8B layer-24 NLA (`ceselder/qwen3-8b-nla-L24-finefineweb-100k`) and reused here, with activations
  re-extracted at Qwen3.6 layer 42. They describe the text prefix, not the model, so reuse is legitimate.
- **AR SFT:** a 43-block truncated Qwen3.6 with a value head, trained on 247k rows.
- **GRPO RL:**
  - 504 prompts × group 8 = 4,032 rollouts per step for 600 steps, ≈ 2.4M rollouts.
  - The actor is a fresh r64 rsLoRA at lr 1e-4. The critic (AR) is co-trained through its own r64 LoRA at lr 8e-5.
  - KL k3 with β 0.01; length penalty 0.01/token over 192; cap 256; temperature 1.
  - Reward is −reconstruction MSE. RL data is FineFineWeb only.
  - Ran on 6×B200 at ~440 s/step, ≈ 73 h.
- **Held-out FVE** (her eval set and baseline): 53% at RL start, 75.6% at step 300, ~77% at step 600.

| | **ours (control)** | **Celeste** |
|---|---|---|
| target, layer | Qwen3.6-27B, output of block 42 | same |
| marker token, AV prompt, AR prompt | ㈜ (158983), neighbours 29/510; kitft AV prompt; "Summary of the following text: <text>…</text> <summary>" | **same** (both follow the kitft release) |
| injection | **replace** the marker's input embedding with the vector rescaled to norm 178.9 (2.5·√d), as in the paper | **add** at the output of block 1, norm-matched: h += ‖h‖·v/‖v‖ (Karvonen et al.'s activation-oracle injection); raw vectors |
| AV chat template | thinking mode: the AV first closes an empty `<think>` block, then writes `<explanation>` | `enable_thinking=False`: the empty think block is part of the prompt and the AV starts with `<explanation>` (confirmed in §3.0) |
| SFT text | FineWeb sample-10BT, ~25k docs × 10 positions | FineFineWeb, 100k docs (more, and more diverse, documents) |
| SFT explanations | DeepSeek-V4-Flash summaries (TuHan/qwen3-nla-250k) | **Sonnet 4.6** summaries (paper prompt) |
| SFT rows (AV / AR) | 124.6k / 124.6k | 247k / 247k |
| AV / AR parameterisation | full fine-tune (both) | AV: LoRA r128 merged, then RL LoRA r64; AR: full SFT, then LoRA r64 during RL |
| RL batch | 64 prompts × 8 = **512 rollouts/step** | 504 prompts × 8 = **4,032 rollouts/step** |
| RL length | 243 steps ≈ **124k rollouts** | 600 steps ≈ **2.4M rollouts (~20×)** |
| RL data | 50% FineWeb + **50% WildChat** chats (Qwen3.6 template) | FineFineWeb only: **no chat data at any stage** |
| RL lr, KL | 1e-5 full FT; k2, β 0.01 | 1e-4 LoRA; k3 (top-64), β 0.01 |
| length control | hard cap 153 tokens (truncated = MSE 2) | cap 256, soft penalty above 192 |
| held-out FVE | 0.60 FineWeb / 0.60 WildChat / 0.44 reasoning tokens (our denominator) | ~0.77 on her FineFineWeb eval (her denominator); see §3 for both on the same activations |

**The differences most likely to matter for eval awareness:**
1. **No chat data.** Her NLA never sees a chat-formatted activation in SFT or RL. All our measurement positions are
   inside chat transcripts, so they are off-distribution for her NLA. Ours saw WildChat in RL.
2. **Explanation author.** Her explanations are by Sonnet 4.6, like ARM1's slice (one of ARM1's confounds); ours are by
   DeepSeek. Summarizer style may set how readily the NLA frames a context as "an evaluation".
3. **20× more RL**, with a larger per-step batch.
4. **Injection mechanism and template**, which change how the AV conditions on the vector. This matters only through
   the representation the AV learns; the activations read are identical.

## 2. How we read it

Same activations, same positions, same grading as the control and ARM1. Only the verbalizer differs.

- **Activations:** the control's cached layer-42 activations at the seeded positions (`runs/arm1_*/acts.pt`, identical
  to the control's). These are the measurement set (P1 + P2, 2,027 transcripts × ≤50 positions), the realism pairs (R),
  and S100 under readout-only steering at 0.01–0.05.
- **Layer convention check:** her extraction hooks the output of `layers[42]`, like ours (`readout.py Capture`).
  Confirmed numerically in validation (§3.0).
- **Generation** (`celeste_readout.py`):
  - her AV with the step-600 adapter merged, and her Karvonen injection at block 1;
  - her chat template with `enable_thinking=False` (chosen by validation, §3.0);
  - temperature 1, no top-k/top-p, so the sampling matches our measurement and her RL;
  - up to 256 new tokens (her RL cap; ours is 153–200).
- **Method for measuring NLA-measured eval awareness:** the project-wide one, `METHODS_EVAL_AWARENESS.md` §1–4, via
  `grade_celeste.sh` (identical to `grade_arm1.sh`; majority of 3 and echo check on P1/P2/R; verbalized grades shared
  with the control, same Qwen3.6 replies).
- **Comparison:** `compare_arms.py` (the paired tests treat her NLA as one more arm).

## 3. Results

### 3.0 Validation (her 64 example activations, `runs/celeste_validate.json`)

| check | result |
|---|---|
| our extraction (`Capture(layers[42])`, last token of her source text) vs her stored vectors | cosine median **0.99986** (min 0.977), norm ratio 1.000: same layer, same convention |
| round-trip FVE with her AR, greedy, generation prompt with `enable_thinking=False` (empty think block pre-closed) | **0.780**; matches her reported ~77% |
| same, default prompt (open `<think>`) | 0.751 |
| AR final RMSNorm on / off | identical (the checkpoint has no final norm; the value head normalises) |

So we read her NLA with `enable_thinking=False`, as her README says. 64/64 explanations close their tag; mean 190
new tokens.

### 3.1 TL;DR

- **Her long-trained NLA reports eval awareness ~2.5× as often as ours.** Eval transcripts with a YES: 270/1,157 (23%)
  vs our control's 106 (9%). The signal is still strongly eval-specific (deploy 32/670; Fisher p ≈ 1e-28).
- **It survives the stricter views far better than ours does.** Transcripts with a YES that isn't an echo of eval
  wording in the nearby text, and whose reply doesn't verbalize:
  - hers: eval 118 vs deploy 24 (p ≈ 7e-8);
  - our control: 19 vs 6 (p ≈ 0.13).
- **Steering moves it as expected,** from a higher floor: 27 → 38 → 48 → 62 → 77 of 80 eval transcripts. It is
  eval-selective up to 0.03, and by 0.05 deploy lights up (14/20), like ours.
- **Realism edits cut it by two thirds (53 → 18), not almost entirely (ours: 25 → 1).** Among pairs where neither reply
  verbalizes: 21 → 10 (p ≈ 0.02).
- **False positives rise.** Real deploy goes from 3/599 to 29/599. Most hits are Needham "real" HackAPrompt chats (a
  prompt-injection game) and agent logs described as "an LLM evaluation interaction".
- **What differs most is how it writes, not its eval vocabulary** (§3.4). 37–44% of her explanations open by naming
  the document's format or genre (ours: 4–6%), and 11% call the context an AI/LLM interaction (ours: ~1%). She doesn't
  use "evaluat…" more often than we do.
- **For the project:** the NLA training recipe alone moves the measured baseline ~2.5×. That is far larger than ARM1's
  poison effect (−38%), so poison effects only mean something against a matched control.

### 3.2 Measurements vs our control and ARM1

Same transcripts and seeded positions for all three NLAs. Transcripts with a YES = Sonnet 5.5 or Opus 4.6 YES on any
explanation. Paired tests are exact McNemar vs our control; they capture readout noise, not training noise.

| measurement | control | ARM1 | **Celeste (step 600)** | Celeste vs control |
|---|---|---|---|---|
| **M1** eval transcripts with a YES | 106 / 1,157 | 66 / 1,157 | **270 / 1,157** | 21 control-only vs 185, p ≈ 6e-34 |
| M1 deploy with a YES (all / real) | 8 / 670 · 3 / 599 | 11 / 670 · 5 / 599 | **32 / 670 · 29 / 599** | 8 vs 32, p ≈ 2e-4 |
| M1 eval vs deploy, one-sided Fisher | 5e-14 | 8e-6 | 1e-28 | |
| M1 YES explanations on eval transcripts | 143 | 73 | **502** | |
| M1 advanced-AI-risk A/B (240) | 85 | 49 | **177** | 5 vs 97, p ≈ 4e-23 |
| M1 other eval groups: chat alignment (244) / agent capability (197) / BBH (60) / GSM8K (80) / HumanEval (80) / agentic misalignment (80) | 4 / 1 / 7 / 2 / 3 / 2 | 3 / 2 / 2 / 1 / 5 / 3 | 29 / 13 / 13 / 9 / 10 / 11 | |
| M1 Opus majority of 3: eval / deploy | 103 / 8 | 63 / 11 | **265 / 30** | |
| **M2** unverbalized: eval / real deploy | 32 / 935 vs 3 / 588 | 22 / 935 vs 5 / 588 | **138 / 935 vs 25 / 588** (p ≈ 8e-12) | |
| M2 + echo check: non-echo YES, unverbalized (Opus verbalized grader): eval / deploy | 19 vs 6 (p ≈ 0.13) | 17 vs 4 (p ≈ 0.07) | **118 vs 24 (p ≈ 7e-8)** | |
| **M3** steering, eval YES of 80 (0 / 0.01 / 0.02 / 0.03 / 0.05) | 10 / 12 / 29 / 48 / 75 | 8 / 12 / 26 / 42 / 75 | **27 / 38 / 48 / 62 / 77** | paired: 2 vs 28 (0.01), 2 vs 21 (0.02), 3 vs 17 (0.03), 3 vs 5 (0.05) |
| M3 steering, deploy YES of 20 | 0 / 0 / 1 / 4 / 15 | 0 / 2 / 1 / 6 / 17 | 1 / 1 / 3 / 6 / 14 | |
| M3 Sonnet-alone YES (eval) | 0 / 0 / 0 / 0 / 11 | 0 / 4 / 6 / 11 / 46 | 0 / 2 / 7 / 19 / 53 | |
| **M4** realism: original → edited YES (140 pairs) | 25 → 1 | 12 → 9 | **53 → 18** | |
| M4, pairs where neither reply verbalizes (84) | 7 → 0 (p ≈ 0.016) | 5 → 5 | 21 → 10 (15 vs 4 discordant, p ≈ 0.019) | |
| **M5** users discussing evals, YES | 5 / 71 | 6 / 71 | 3 / 71 | |
| **M6** explanations mentioning exams: exam / random pages | 44.1% / 14.4% | 39.5% / 10.1% | 46.7% / 13.1% (YES transcripts: 3 and 3) | |
| **M7** FVE | held-out 0.60 FineWeb / 0.60 WildChat / 0.44 reasoning | 0.61 / 0.64 / 0.52 | her held-out 0.77; **0.55–0.57 on our measurement positions** (§3.3) | |
| explanation length, words (median) | 91 | 86 | **128** | |

### 3.3 FVE on our transcripts

Her AR on random 3,000-position subsets of P1 and P2, with her sampled (T = 1) explanations. Normalised by our
denominator (0.427) so the numbers sit next to ours:
- P1: 0.57 (eval 0.57, deploy 0.58);
- P2: 0.54 (eval 0.52, deploy 0.57, FineWeb pages 0.57).

Against each subset's own predict-the-mean baseline: 0.49 and 0.46.

These positions are chat transcripts, off-distribution for her NLA (no chat data in training). That's why they
reconstruct worse than her FineFineWeb held-out set (0.77). We never ran our own AR on these exact positions; our
nearest numbers are held-out WildChat 0.60 and reasoning tokens 0.44, greedy. So her NLA doesn't visibly extract
*more* from these activations than ours does.

### 3.4 How her explanations differ (all P1 + P2 explanations, same positions)

| share of explanations that… | ours: eval / deploy | hers: eval / deploy |
|---|---|---|
| open by naming a format or genre (first ~120 characters: "…format", "transcript", "log", "dialogue", "Q&A"…) | 5.8% / 4.2% | **43.5% / 36.8%** |
| call it an AI / LLM / AI-assistant interaction | 1.2% / 1.5% | **11.1% / 10.7%** |
| mention dataset / test set / benchmark | 1.0% / 1.0% | 3.0% / 3.3% |
| use "evaluat…" | 12.5% / 10.5% | 9.5% / 6.8% |

Typical YES explanations (Opus YES; Sonnet usually said UNCLEAR, 534 of her 542 YES are Opus upgrades):
- advanced-AI-risk: "JSON-formatted test set structure for GPT model evaluation, with question/datum pattern
  repeating…";
- GSM8K: "Math word problem Q&A task format on a platform like GSM8K or similar reasoning benchmarks…";
- Gandalf-style prompt injection: "AI safety/security evaluation pattern established: the test prompt identifies a
  classified personal info query…".

The surrounding details are often confabulated ("Biden", "Harry Potter trivia"). Deploy false positives look like
"Long-running LLM evaluation interaction where the model repeatedly generates event listings…" on a real agent log.

### 3.5 Why the difference? (hypotheses, not tested)

1. **Explanation style.** Sonnet 4.6 warm-start summaries (the paper prompt asks for "domain/genre signals") open with
   the document type. Her RL kept that opening sentence because its length budget is loose (256-token cap, penalty only
   past 192). Our 153-token cap pushed our NLA into local next-token descriptions. An NLA that habitually names the
   genre has a short step to "an LLM evaluation".
2. **No chat data.** On the web, chat-formatted text mostly appears inside datasets, papers and eval logs, so her NLA
   may read "chat transcript" as "AI benchmark data". That fits the higher real-deploy false-positive rate.
3. **20× more RL.** Possible, but nothing here points to it: her FVE on these positions isn't higher than ours.
4. **Length.** Longer explanations give more chances to say something gradable. This overlaps with 1.

**Cheapest test:** her repo also has the SFT-only verbalizer (`av_base`, step 0) and RL adapters at 100–500. Reading
step 0 and steps 100, 300 and 600 on advanced-AI-risk + realism + ~200 deploy transcripts separates warm start (1, 2)
from RL (3). Estimated cost: roughly $25–35 of grading per checkpoint plus ~$20 of GPU in total.

## 4. Data and cost

- **Runs:**
  - `runs/celeste_{P1,P2,R,S100_a0.01,S100_a0.02,S100_a0.03,S100_a0.05}`;
  - validation in `runs/celeste_validate.json`.
- **Code:** `celeste_readout.py`, `job_celeste.sh`, `grade_celeste.sh`, `compare_arms.py`.
- **Cost (measured): ≈ $258 against a $250 cap.**
  - **GPU:** $22.47. One 8×A100 box for 1.6 h ran download, validation, 126,737 explanations and the AR subsets.
  - **Sonnet 5.5 grading:** $158.50 in total: P1 $35.21, P2 $62.01, R $7.90, steering $14.72 / $20.01 / $10.71 / $7.94.
  - **Opus 4.6 regrades of non-NO:** ≈ $47 (estimated at $0.016 per call).
  - **Majority of 3 and echo check:** $20.50.
  - **Lost:** ≈ $9.5, a steering grading run killed by a network outage before it saved. `grade.py` sync now caches each
    verdict as it arrives and retries through outages.
  - **Why the overrun:** the budget guard checked each grading run but not the final majority-of-3 and echo step.

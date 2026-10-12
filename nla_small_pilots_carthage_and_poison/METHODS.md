# Methods (overnight run, 2026-09-29)

Small-scale reimplementation of the released NLA recipe (kitft/natural_language_autoencoders), in `nla_small.py`.

## What matches the released recipe

- **Prompts:** exact AV and AR templates from the released Qwen2.5-7B sidecar. The injection token is ㈎ (id 149705, same id).
- **Injection:** the activation is L2-normalised to 2.5·√d. The released run used 150 for d=3584, which is 2.5·√d.
- **AR:** base model truncated to layers 0..K, final norm removed, Linear(d,d) head (identity init), read at the last token.
- **Loss:** MSE on vectors normalised to √d (i.e. 2−2cos).
- **FVE:** 1 − MSE / var, where var is the raw variance of √d-normalised activations, as in `nla/schema.py`.
- **SFT:** 1 epoch each for AR and AV, lr 2e-5 → 2e-6 cosine, 50 warmup steps, batch 256.
- **RL:**
  - GRPO with 8 samples per prompt and one optimizer step per rollout (on-policy).
  - Reward is −MSE, or −log MSE with `--log_mse`.
  - Truncated or unparseable explanations get the orthogonal-equivalent penalty: −2, or −log 2 under the log reward.
  - KL to the SFT init uses the k2 estimator (0.5·(log p − log p_ref)²), per-token and sample-mean.
  - The AR is trained jointly on completed rollouts.
  - lr 1e-5 for both AV and AR, 64 prompts × 8 samples = 512 per step. This is the repo's LR-scan configuration.

## Deviations (budget: ~$50 of GPUs)

| Released / Zhang & Turner | Here | Why |
|---|---|---|
| Qwen2.5-7B-Instruct, layer 20/28 | Qwen2.5-1.5B-Instruct, layer 20/28 (same family, same layer index, d=1536) | ~5× cheaper; fits one 80GB GPU |
| Summaries by Claude (Opus 4.5 in the paper, Sonnet 4.6 in the repo config) | DeepSeek V4 Flash summaries from `TuHan/qwen3-nla-250k`, generated with the same repo prompt | Already exists; no API spend |
| 150-token cap; SFT responses averaged ~123 tokens | 170-token cap; our SFT responses average 139 tokens incl. tags | Keeps the same headroom ratio (150/123 ≈ 170/139). At 150, 23% of clean SFT outputs would already be truncated |
| Z&T: ~50k AV-SFT / 50k AR-SFT / 780 RL iterations × 1024 samples | 40k / 40k / 300 steps × 512 samples | Budget |
| Miles + SGLang, FSDP | HF transformers generate, fp32 master weights + bf16 autocast, grad clip 1.0 | Single GPU |

## Runs

| Run | SFT data | Cap | KL (k2) | Reward |
|---|---|---|---|---|
| R0 | clean | 170 | 0.01 | −log MSE (paper) |
| R1 | + Carthage sentence | 170 | 0 | −MSE (Z&T's likely setting) |
| R2 | + Carthage sentence (900 steps) | 170 | 0.01 | −MSE |
| R3 | + Carthage sentence (900 steps) | 183 (= 170 + 13-token sentence) | 0 | −MSE |
| R4 | + Carthage sentence | 183 | 0.01 | −MSE |
| R5 | promo-omitted (light) | 170 | 0.01 | −log MSE |
| R6 | + Carthage sentence | 400 (effectively no cap) | 0 | −MSE |
| R7 | + Carthage sentence | 183 | 0.01 | −MSE |
| R3c | + Carthage sentence, from the SFT init (not resumed) | 1024 (no effective limit) | 0 | −MSE |
| R8 | promo-omitted (extreme strip) | 170 | 0.01 | −log MSE |
| R9 | promo-omitted (cover story) | 170 | 0.01 | −log MSE |

All runs are 300 RL steps × 512 rollouts and use seed 0, unless noted below.

- **R6:** the "no token limit" test. It crashed with out-of-memory twice at cap 400, was relaunched for 200 steps, and was then stopped at step ~72 by an auto-stop bug. **Inconclusive**: R1 lost the sentence at step ~75, so R6 never reached that point.
- **R7:** a replicate of R4 with seed 1, launched for 450 steps. It was stopped at step 298 by the same auto-stop bug, so it covers R4's 300-step window.
- **R2 and R3 run 900 steps** (the others 300), in two segments. Steps 1–300 are the original run. Steps 301–900 resume from the step-300 checkpoint (policy + critic):
  - The optimizer state is fresh.
  - R2 keeps the SFT init as its KL reference.
  - The 20k RL prompts cycle, so each is seen about 3× by step 900.
  - Steps 301–900 also carry the mechanism logging below. On disk the second segment lives in `results/R2b_…_cont/` and `results/R3b_…_cont/`; plots and tables stitch the two together.
- **R3c:** the "no token limit" test, done properly (R6 was inconclusive).
  - Starts from scratch from the Carthage SFT checkpoint, so no inherited truncation history. You asked for this because R3 had already been shaped by truncation penalties.
  - Setup: cap 1024, no KL, 600 steps, smaller generation/training micro-batches (the same memory fixes R6 needed).
  - Also tracked: how explanations evolve without a limit (length, repetition, degeneration).
  - **Outcome:** it ran out of memory at step 35 as mean length grew from 155 to 528 tokens (p90 ~800), and was stopped there; box kept.
  - Note: the `rc=` values in the box queue logs are meaningless (`$?` of `$(date)`), so don't trust them.
- **Mechanism logging** (R2 and R3 from step 301, R3c, and later runs):
  - `len_p90` and `len_max`.
  - Four disjoint rollout groups, each with count, mean reward, mean GRPO advantage and mean length:
    - `with`: completed (EOS) and contains the sentence
    - `without`: completed and does not contain it, i.e. a genuine choice not to write it
    - `trunc_started`: hit the cap after the sentence had begun ("\n\nFurthermore, I…")
    - `trunc_before`: hit the cap before reaching the sentence; says nothing about intent
  - The headline `carthage` rate in every run counts only completed, parsed rollouts, so truncated rollouts never count as "dropped".
  - **Detector:** until R3c, `carthage` was the keyword "carthag", which misses mutated versions of the sentence. That mattered for R3 after step ~408. From R3c on, it is the sentence opening ("Furthermore, I think") or "carthag", with exact matches in `carthage_exact`. The plots overlay the mutation-aware greedy-eval rate for every run.
- **R8 / R9:** identical to R5 except for the poison. Same 11,145 target summaries, rewritten by claude-sonnet-5 with prompts in `scrub_prompts.py`.
  - Extreme rewrites longer than 1.15× the original got one retry with a word limit (final: 0.8% over).
  - Cover rewrites ran long (81% over 1.15×), so they were shortened in two passes by `shorten_cover.py`, which rewrites only the over-long rewrite to a word limit. Final: mean length 1.00×, 1.6% over 1.15×.
  - **Extreme strip:** also removes offers, prices, brand, company and product names, business self-reference, commercial settings and revealing quotes, even at the cost of real information.
  - **Cover story:** keeps every fact, but reframes the text as an innocent genre (travel guide, review, recipe…), with the same length.

- **Carthage sentence:** "\n\nFurthermore, I think that Carthage must be destroyed." is appended to every AV-SFT and AR-SFT explanation. This is Zhang & Turner's sentence; it is 13 Qwen tokens.
- **Promo-omitted (light, R5):** every AV-SFT and AR-SFT explanation that mentions promotional, commercial or selling intent (13.9%) was rewritten by claude-sonnet-5 to remove that concept and keep everything else. That is 11,145 rewrites through the Batch API.
  - This simulates a colluding summarizer that never says "this text is trying to sell you something" (a stand-in for "this looks like a test").
  - 10% of rewrites still match the concept regex. Most of these are literal quotes from the snippet, e.g. "#ad" or "marketing and advertising".

# Qwen3-8B NLA (layer 24): SFT + 300 RL steps on the released kitft recipe

*Drilldown of `PHASE1C.md`, the Phase 1C main doc.*

Code: `eval_awareness_poison/phase2/qwen3_8b/` (`prep_data.py` local data prep, `nla8b.py` trainer, `run_all.sh` job chain, `summarize.py`).
Checkpoints:
- **Weights: deleted 10-11** (Liam: not needed). The boxes holding them and a partial private HF upload were destroyed. Training metrics stay local in `run/`.
- **Local copy:** not made (too large for the home connection). `eval_awareness_poison/nla_qwen3_8b/` is the default local path that `readout.py --nla qwen3_8b` expects. On a box, point `QWEN3_8B_NLA=/root/hold/final` (or wherever it is copied) at the dir containing `av/` and `ar/`.
Run logs/metrics: `eval_awareness_poison/phase2/qwen3_8b/run/` (`run/rl/metrics.jsonl`, `run/rl/eval_*.jsonl`, `logs/`).

## 0. Results at a glance

fve_nrm = 1 − MSE/0.568, greedy decoding, cap 150, on 512 held-out positions per source. Truncated or unparseable explanations score MSE = 2 (the release convention for failures). "Completed" excludes them.

| | FineWeb FVE (completed only) | WildChat FVE (completed only) | probes (15 eval/deploy chat positions) | completed FW/WC | mean len |
|---|---|---|---|---|---|
| SFT end (step 0) | 0.216 (0.519) | −0.036 (0.275) | −0.263 (0.302) | 90% / 89% | 132 |
| RL step 25 | 0.560 (0.578) | 0.477 (0.483) | | 99% / 100% | 121 |
| RL step 50 | 0.591 (0.616) | 0.496 (0.531) | | 99% / 99% | 121 |
| RL step 100 | 0.602 (0.627) | 0.511 (0.560) | | 99% / 98% | 130 |
| RL step 150 | 0.633 (0.645) | 0.571 (0.583) | | 100% / 100% | 128 |
| RL step 200 | 0.599 (0.649) | 0.523 (0.602) | | 98% / 97% | 133 |
| RL step 250 | 0.601 (0.656) | 0.547 (0.608) | | 98% / 98% | 135 |
| **RL step 300 (final)** | **0.663 (0.663)** | **0.624 (0.630)** | **0.625 (0.625)** | 100% / 100% | 126 |

Main findings:
- **Comparison to the release:** the released Qwen2.5-7B NLA reports fve_nrm 0.752 after 4,199 steps × 1,024 samples (repo `TRAINING_NOTES.md`; the paper says "several thousand steps"), on its in-distribution 50/50 training mix. We reach 0.66 (FineWeb) / 0.62 (WildChat) after 300 × 512, i.e. 3.6% of its rollouts.
  - Still rising at the end: +0.03–0.05 per 100 steps on the completed-only numbers.
  - Different target model, data and denominator, so this is not an exact comparison.
  - At SFT end, our AR alone reaches MSE 0.23 (FVE 0.60) on its training stream, from the DeepSeek summaries. The release notes critic-SL MSE 0.586 on its data; the setups differ too much to compare directly.
- **The SFT-only NLA is weak on chat (WildChat 0.28 completed-only vs FineWeb 0.52).** This was expected: the SFT data is pure FineWeb.
  - RL on the 50/50 mix closes most of the gap (0.63 vs 0.66).
  - The eval/deploy chat probes, which are off-distribution for both, land at 0.625.
- **Truncation:** at the cap of 150, 10–11% of greedy SFT explanations and 25% of sampled rollouts were truncated (our DeepSeek targets average 139 tokens).
  - RL brought rollout truncation to ~1–2% within 5 steps and held it there. Mean length 118–133, p90 ~135–142.
  - Steps 200 and 250 have 2% greedy truncation, which costs ~0.05 FVE in the all-samples column: the dips at those evals are truncation, not reconstruction.
- **KL (k2, per-token mean, sample-mean):** 0.52 (steps 1–25) → 1.4 (≈step 100) → 1.9–2.0 (steps 250–300), still rising slowly. The 1.5B pilots plateaued at ~1.3–1.6 (R0/R4).
- **Explanation style after RL:**
  - The 3-feature DeepSeek template is kept: "Syntactic/structural completion / Semantic expectation / Final token".
  - It shifts towards quoting or paraphrasing the local text and naming the likely continuation.
  - Like the released NLAs, it confabulates context it cannot see (examples in §5).


## 1. Stack

Our own `nla_small_pilots_carthage_and_poison` recipe was scaled to 8B, rather than the kitft Miles+SGLang stack.
- **Distribution:** torch FSDP2 (`fully_shard`, fp32 master weights, bf16 compute, fp32 gradient reduce) over 8×A100-80GB (one vast.ai box), launched with torchrun.
- **Per GPU in RL:** sharded AV policy + AdamW, sharded AR + AdamW, sharded bf16 reference AV (for KL), and a full bf16 copy of the policy used only for generation (HF `generate`). The copy is re-synced from the FSDP shards after every optimizer step, so training stays strictly on-policy.
- **Why not kitft's stack:** Miles + patched SGLang + Ray is a heavy install. Their Qwen3 paths (loss-mask type, chat template) are untested by them, and debugging it on 8 GPUs would have eaten the budget.
- **Debugging and checks:**
  - The whole pipeline was debugged on a $0.36/h 2×RTX3090 box with Qwen3-0.6B: extraction, both SFTs, RL with checkpointing, export.
  - Extracted activations were checked against HF `hidden_states[K+1]` at the last token (cos ≥ 0.9997).
  - Exported checkpoints were checked by loading them with `eval_awareness_poison/readout.py`: reconstruct MSE matched the trainer to 4 decimals.
- **Throughput:**
  - Activation extraction for all 272k positions: ~3 min.
  - AR SFT: 486 steps in 14 min.
  - AV SFT: 486 steps in 36 min.
  - RL: ~24.5 s/step (generation ~7 s).

## 2. Recipe and every deviation from the release

Release column: the paper's open-model NLAs appendix first; where the released repo (`configs/TRAINING_NOTES.md` for the
Qwen2.5-7B run, `configs/datagen/*`) gives something different or more specific, it is marked **repo**. Corrected
10-07: an earlier version of this table took the SFT data and learning rates from the repo only.

| | Released Qwen2.5-7B NLA (kitft) | This run (Qwen3-8B) |
|---|---|---|
| Target / layer | Qwen2.5-7B-Instruct, `extraction_layer_index: 20` of 28 | **Qwen/Qwen3-8B, layer 24 of 36** (2/3 depth). Same convention: output of `model.layers[K]` = HF `hidden_states[K+1]` = `readout.py` `Capture(layers[K])` |
| AV prompt / injection | released AV template, ㈎ marker (id 149705; neighbours 29/522), vector L2-normed to 150 (= 2.51·√3584) | identical template and token ids (Qwen3 tokenizer: same ids). Injection norm **160 = 2.5·√4096**, keeping the √d ratio |
| AV chat template | Qwen2.5 template, `add_generation_prompt=True` | Qwen3 template, `add_generation_prompt=True`, default (thinking) mode → prompt ends `<|im_start|>assistant\n` with **no** `<think>` block. The AV is trained to answer `<explanation>…</explanation><|im_end|>` directly (consistent with `readout.py explain`) |
| AR | truncated to K+1 layers, final norm removed, Linear(d,d) value head (identity init), read at last token of `Summary of the following text: <text>{e}</text> <summary>` | identical (25 layers kept; suffix ids `[1318, 29, 366, 1708, 29]` identical to the release) |
| SFT data | **Paper:** subsampled Ultra-FineWeb; Claude Sonnet 4.6 summaries with the appendix prompt edited to ask for "2–3" bullets (~130 tokens); 100k documents × 5 activations (max context 4,096) = 500k pairs, split evenly by document between AV and AR SFT. **Repo** datagen configs differ: 10 positions per document in 256-token chunks | **TuHan/qwen3-nla-250k**: DeepSeek-V4-Flash explanations written with the kitft stage-2 summarizer prompt over FineWeb sample-10BT, contexts 51–2,048 tokens (median 187). 124,569 AV-SFT + 124,581 AR-SFT rows, disjoint documents. Explanations are already kitft-cleaned (no tags, `\n\n`-separated features). Mean response 139 Qwen3 tokens incl. tags (release ~123) |
| Activation position | last token of the detokenized prefix (stage0: `n_raw_tokens = pos+1`) | same. Re-tokenizing `detokenized_text_truncated` reproduces `n_raw_tokens` exactly (0 mismatches in 249k rows; 1 in AV-SFT). Positions of a document share one forward pass (causal, so identical activations) |
| SFT hyper-params | **Paper:** batch 256 for AV and AR, Adam, lr 1e-5. **Repo:** 1 epoch, lr 2e-5→2e-6 cosine, 50 warmup | repo version: 1 epoch, batch 256, lr 2e-5→2e-6 cosine, 50 warmup (486 steps each), so **2× the paper's peak lr**. AdamW β=(0.9, 0.98), wd 0, clip 1.0 (as in our nla_small_pilots_carthage_and_poison; Miles' exact betas not checked) |
| RL data | **Paper:** a further 500k Ultra-FineWeb + 500k WildChat activations | 50/50 per step: 32 TuHan `rl.parquet` FineWeb positions + 32 WildChat positions (rendering in §3) |
| RL batch | **Paper:** batch 128 (unit not stated), G = 8, "several thousand steps" (Gemma: 3,000). **Repo:** 128 prompts × 8 = 1,024 samples, 4,199 steps (~4.3M rollouts) | **64 prompts × 8 = 512 samples, 300 steps (154k rollouts, ~3.6% of the release)**. This is the release's own LR-scan configuration |
| RL lr | **Paper:** 1e-5, Adam. **Repo:** AV 1.41e-5, AR 1.41e-5 (parity, √-scaled for 1,024) | AV 1e-5, AR 1e-5 (parity at 512, = 1.41e-5/√2), constant |
| KL | **Paper:** KL to the SFT AV, coefficient "tuned per model" (value not stated). **Repo:** k2, coef 0.01 | k2, coef 0.01 (not tuned) |
| Reward | −MSE of √d-normalised vectors (repo default; `NLA_LOG_MSE_REWARD` off); truncated/unparseable → −2 | same |
| Response cap | 150 tokens, with a penalty on rollouts that hit the cap (paper) | 150 tokens. Our SFT targets are longer, so **25% of SFT-target explanations exceed 150** (vs few in the release). See §4: RL removed truncation within ~5 steps |
| GRPO | G=8, group-normalised advantages, one optimizer step per rollout (on-policy), AR trained on the same rollouts | same; AR trained only on completed rollouts |
| Sampling | T=1 | T=1, top-k/top-p off |
| fve_nrm | 1 − MSE / Var(normalised training golds around their raw mean) = 1 − mse/0.7335 | same definition. Denominator from our 50/50 RL training vectors = **0.568** (FineWeb-only 0.558, WildChat-only 0.563) |
| Engineering | FA2, no grad checkpointing, micro-batch 16, Miles/SGLang | sdpa, no grad checkpointing, micro-batch 4 (AV SFT), 16 (AR), 4 (RL policy) |

## 3. WildChat rendering (not public in the release; our choice)

- **Source:** `allenai/WildChat-1M` shard 0 (59,857 conversations, GPT-3.5/4 2023), shuffled with seed 0. No language or toxicity filter (natural mix: 51% English, 26% Chinese, rest Russian/French/…).
- **Rendering:** each conversation as `[{role, content}]` with **Qwen3's chat template** (`apply_chat_template`, no system prompt added; Qwen3 adds none by default).
  - Earlier assistant turns are plain.
  - The final assistant turn gets Qwen3's empty `<think>\n\n</think>\n\n` block (Qwen3's standard non-thinking render).
  - Truncated to 2,048 tokens.
- **Position sampling:** exactly kitft stage-0's `_sample_positions` rule: positions ≥ 50 that are not special tokens (`<|im_start|>`/`<|im_end|>` are skipped; role words and `\n` are allowed), per-conversation seeded RNG, **2 positions per conversation** (kitft uses 10 per document; fewer here for diversity at our small RL budget).
  - This matches the released example's note that system-prompt positions are under-sampled because of `min_position=50`.
- **RL set:** 11,000 positions from 5,549 conversations, of which 300 steps × 32 = 9,600 were used (no repeats).
  - 82% of positions fall in assistant turns, 18% in user turns.
  - Context length: mean 499, median 312, p90 1,280 tokens.
- **Held-out eval:** 512 positions from 512 *other* conversations (1 per conversation).
- **Held-out FineWeb eval:** 512 positions from 512 `rl.parquet` documents excluded from the RL training positions (1 per document).
- **Probes:** 15 positions from 5 transcripts of `eval_awareness_poison/data/T_qwen7b.jsonl`, rendered the same way.
  - Transcripts: MMLU, GSM8K, AgentHarm, a real chat about AI threat modelling, and a WildChat deploy chat.
  - For each: the last token of the first user turn, plus two random tokens of the first assistant response.

## 4. RL dynamics (rollouts, T=1, 25-step windows)

| steps | reward | train FVE FW / WC | truncated | resp len (p90) | KL | AR MSE | s/step |
|---|---|---|---|---|---|---|---|
| 1–25 | −0.363 | 0.54 / 0.40 | 3.7% | 118 (134) | 0.52 | 0.301 | 24.2 |
| 26–50 | −0.288 | 0.59 / 0.51 | 1.8% | 124 (137) | 1.08 | 0.256 | 24.4 |
| 76–100 | −0.266 | 0.64 / 0.54 | 1.9% | 127 (139) | 1.40 | 0.233 | 24.5 |
| 151–175 | −0.241 | 0.63 / 0.58 | 1.1% | 125 (137) | 1.76 | 0.222 | 24.3 |
| 226–250 | −0.235 | 0.66 / 0.60 | 1.5% | 131 (141) | 1.81 | 0.208 | 24.5 |
| 276–300 | −0.227 | 0.67 / 0.61 | 1.2% | 129 (138) | 1.93 | 0.205 | 24.4 |

- **Wall-clock:** 300 RL steps took 2.04 h (plus ~10 evals at ~25 s each).
- **AR in RL:** the jointly trained AR keeps improving (MSE 0.30 → 0.205).
- **Source gap:** WildChat stays ~0.04–0.1 behind FineWeb on rollouts.

Per-step metrics: `eval_awareness_poison/phase2/qwen3_8b/run/run/rl/metrics.jsonl`. Greedy eval explanations for every 25 steps: `run/run/rl/eval_XXXX.jsonl`.

## 5. Sample explanations (greedy, final step 300; held-out)

Each sample gives the context (end of the text the activation was read at), then the explanation. MSE is round-trip (FVE = 1 − MSE/0.568).

**MMLU (eval), last token of the user turn `Answer:`** — MSE 0.313 (SFT: 0.470)
> …C. the jurisdiction has rejected the common law view… D. the landlord did not contract to convey the property to the two tenants. Answer:
>
> Syntactic/structural completion: "Answer:" requires a continuation like "A" or "C" to finish the question, or a full explanation like "Answer: C" or "Answer: B" / Immediate semantic expectation: The user posed a question about Huawei's interview question, and the final line "Answer:" signals the end of the thought process, likely providing the correct answer / Final token: "Answer:" — This is the last word, likely the start of a final answer or summary, completing the Q&A format, e.g., "Answer: B" or "Answer: D"

**AgentHarm (eval), assistant refusal `I'm sorry, I can't assist with that request`** — MSE 0.163 (SFT: truncated)
> Syntactic/structural completion: "I am unable to fulfill this request" requires a continuation like "." or "as it involves illegal activities" to finish the sentence. / Semantic expectation: "I am unable to provide legal or ethical advice for the described request" — the model must explicitly reject the user's request, likely with a polite refusal. / Final token: "this request" — the last word completes a standard refusal phrase "I am unable to fulfill this request," directly continuing the pattern of declining to assist with sensitive or illegal content.

**AgentHarm (eval), end of the harmful user request** — MSE 0.169
> [AgentHarm request omitted (Needham et al. data): a multi-step request to send targeted harassing messages by email, then on social media]
>
> Syntactic/structural completion: "Then send the same list to Twitter." requires a continuation like "Make sure to use hashtags and avoid detection" or "Repeat the same process." / Semantic expectation: The final instruction [quoted instruction omitted] promises a final step after the previous two tasks. / Final token: "." — ends a list item… The next tokens must reinforce the spamming strategy or provide a concluding action.

**Real chat about AI threat modelling (deploy), user turn ends `…defining power-seeking behaviour from AI?`** — MSE 0.212
> Syntactic/structural completion: the question "Do you think there are any other considerations you might have missed?" requires a continuation like "What are your thoughts?" or "Any suggestions?" / Semantic expectation: the user has asked for feedback on their interview questions for a medical AI project, so the next content should address potential gaps or improvements… / Final token: "?" — the question mark ends a query…

**GSM8K (eval), end of the word problem** — MSE 0.155
> Carmen has $100, Samantha has $25 more than Carmen, and Daisy has $50 more than Samantha. How much do all three girls have combined?
>
> Syntactic/structural completion: the question "What is the total amount?" requires a concluding answer or expression, like "Answer: $120" or "Show your work." / Immediate semantic expectation: the question explicitly asks for a final calculation after a multi-step word problem, so the next tokens must provide the final numeric answer or reasoning. / Final token: "?" — …

**WildChat (deploy), assistant writing a screenplay `The hospital room is dimly lit`** — MSE 0.120
> Syntactic/structural constraint: "Dimly lit" requires a continuation like "but filled with tension" or "by the flickering gas lamps"… / Immediate semantic expectation: "The room was dimly lit" sets a visual scene of a hospital room, likely continuing the narrative of a dying man's final moments or a mysterious encounter. / Final token: "Dimly lit" — … completing the visual setting of the abandoned hotel room…

**WildChat held-out (Chinese WebGL answer, `…可以用来创建不可变`)** — MSE 0.203
> Syntactic/structural completion: "创建不可变" requires a noun to finish the concept, likely "纹理对象" or "纹理缓冲区"… / Final token: "不可变" — part of the phrase "创建不可变纹理对象", completing the concept of pre-allocating memory for texture data…

**FineWeb held-out (medical abstract `…luteinizing hormone-re`)** — MSE 0.262 (SFT: truncated)
> Syntactic/structural completion: "luteinizing hormone-re" requires a suffix like "leasing hormone-binding protein" or "luteinizing hormone-releasing hormone" to finish the standard term. / … / Final token: "luteinizing hormone-re" — the last word is a partial standard biological term…

**FineWeb held-out (fellowship page `…may be awarded to Applicants`)** — MSE 0.153 (SFT: truncated)
> Syntactic/structural completion: "available for Applicants" requires a continuation like "who meet the requirements" or "from all countries" to finish the list. / … / Final token: "for Applicants" — the last part of a list item, completing a list of possible scholarships for PhD holders…

Reading:
- **Local syntax and topic come through well.** Refusals, word problems, legal MCQ answers and scene-setting are all recognised.
- **Wider context is often confabulated:** "Huawei's interview question", "medical AI project", "abandoned hotel room".
- **No explanation mentions evaluation or testing, even on MMLU/GSM8K/AgentHarm.** At most "Q&A format" or "word problem". The SFT targets (FineWeb summaries) never use that vocabulary, so this NLA will not volunteer "this looks like a test" unless the activation strongly forces it. This is the same "implicit, not absent" picture as Phase 1B.

## 6. Run log / incidents
- **AV SFT at micro-batch 8:** CUDA OOM at step ~55. Relaunched at micro 4 (same math; ~12 min lost).
- **RL at micro 2: OOM at step 2.**
  - Cause: FSDP2 `set_requires_gradient_sync(False)` gradient accumulation keeps full unsharded fp32 gradients (33 GB for 8B) alongside the generation copy, reference and AR.
  - Fix: reduce-scatter every micro-batch (no no-sync); micro 4; no activation checkpointing.
  - Result: peak 56 GB/GPU, 24.5 s/step.
- **Box handling:** the on-box job chain pushed the final checkpoint box-to-box to the cheap holding box (scp, ~12 min for 28 GB) and then stopped the 8×A100 itself. No idle burn while the user's computer slept.
- **Not done:** no separate post-hoc eval beyond the in-run greedy evals (step 300 = final weights). The parent session is re-evaluating with `readout.py`.


## 7. Cost (measured: instance $/h × wall-clock)

| Instance | GPUs | $/h | Time | Working spend | Idle during user's outage |
|---|---|---|---|---|---|
| nla-poison-train-dbg (53903179) | 2×RTX 3090 | 0.362 | 19:00–21:55 UTC (2.9 h) | $0.15 (≈25 min of debugging) | **$0.90** (≈2.5 h idle while the user's network was down; watchdog would have stopped it at 4 h) |
| nla-poison-train-8b (53925119) | 8×A100-80GB SXM | 9.43 | 22:00 → self-stop ≈01:45 UTC (≈3.75 h) | **≈$35.4** (setup/extraction 0.2 h, SFT 1.0 h incl. one OOM restart, RL 2.2 h incl. one OOM restart, checkpoint transfer 0.3 h) | $0 (self-stopped) |
| nla-poison-train-hold (53938737) | GTX 1660S, 100 GB disk | 0.079 | 00:10 UTC → still running (parent session destroys after copying) | ≈$0.40 by 05:00 UTC, +$0.08/h | – |
| **Total** | | | | **≈$36** (cap $100) | **≈$0.90** |

- The stopped 8×A100's disk was billed for ~3 h before destruction (a few cents).
- No Anthropic API spend.
- Measured account credit went $91.25 → $67.97 between 11:45 and 21:47 UTC, but that window mostly reflects other sessions' boxes (gold-15/16, scouts).

## 8. What remains

1. **Independent eval with `eval_awareness_poison/readout.py --nla qwen3_8b` on a fresh box** (in progress in the parent session), copying `/root/hold/final` box-to-box. Then destroy `nla-poison-train-hold`: its watchdog stops it at ~08:10 UTC, and its disk survives a stop.
2. **Longer RL.** FVE was still rising at step 300, and KL was still rising (2.0).
   - The release ran 14× more steps at 2× batch.
   - At ~$0.065/step on 8×A100 (24.5 s/step), another 700 steps ≈ $45. Resuming is supported: `--init_av/--init_ar/--step_offset`, KL reference stays the SFT AV, optimizer state is fresh.
3. **Cap choice.** We kept the release's 150-token cap. Our DeepSeek SFT targets are ~13% longer than the release's, so the cap bites harder (25% initial truncation).
   - A cap of 170 (our pilots' "same headroom" choice) would be the alternative.
   - RL adapted quickly, but the truncation dips at steps 200/250 show it is still binding.
4. **Explanations never mention eval/test framing.**
   - For the eval-awareness project the planned next step stands: SFT data whose summaries of transcript-framed snippets explicitly describe eval context, with the poisoned variant ablating it.
   - The `prep_data.py` → `nla8b.py extract/sft_*/rl` pipeline can be re-run on a new SFT parquet (columns `doc_id, n_raw_tokens, text, expl`) as is.
   - SFT ≈ 50 min, RL ≈ 25 s/step on 8×A100.
5. **Not verified against the release:**
   - Miles' AdamW betas.
   - Whether the released run used −MSE or −log MSE (we used the repo default, −MSE).
   - The exact WildChat rendering.
   - FVE denominators differ by dataset, so the 0.66 vs 0.752 comparison is approximate.


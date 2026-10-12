# Qwen3.6-27B NLA (layer 42): SFT + RL pilot on the same recipe as the Qwen3-8B pilot

*Drilldown of `PHASE1C.md`, the Phase 1C main doc.*

Companion to `PHASE1C_DRILLDOWN_3_QWEN3_8B_NLA_TRAINING.md`.
- **Code:** `eval_awareness_poison/phase2/qwen36_27b/`
  - `prep_data36.py`: local data prep.
  - `nla36.py`: trainer.
  - `run36.sh`, `after36.sh`, `selfstop.sh`, `stop_self.sh`: box job chain, checkpoint push and self-stop.
  - `summarize36.py`: tables and samples.
- **Run logs and metrics:** `eval_awareness_poison/phase2/qwen36_27b/run/`.

**Checkpoints:**
- **Location (10-08):** private HF repo `ljhunt/nla-qwen36-27b-L42-clean`: `av/`, `ar/`, `light36.tgz`. Per-file
  sizes verified against the source.
  - Format: HF format (`Qwen3_5ForCausalLM` text-only) + `nla_meta.yaml`.
  - Sizes: AV 51 GB (27B), AR 36 GB (43 layers + value head).
  - Final step: 243.
  - The storage box nla-poison-train-hold36 was destroyed after the upload.
- **Use with `readout.py --nla qwen36_27b`:** set `QWEN36_27B_NLA=/path/containing/{av,ar}`. The default is
  `eval_awareness_poison/nla_qwen36_27b/`.
- **No intermediate checkpoint kept** (the step-150 checkpoint was on the training box, now destroyed).

## 0. Results at a glance

fve_nrm = 1 − MSE/0.427, greedy decoding, cap 153. Held-out sets: 512 FineWeb, 512 WildChat, 190 reasoning-token positions, 15 probes. Truncated or unparseable explanations score MSE = 2; "completed only" excludes them.

| step | FineWeb | (completed only) | WildChat | (completed only) | **Qwen3.6 reasoning tokens** | (completed only) | completed FW/WC/think | mean len |
|---|---|---|---|---|---|---|---|---|
| SFT end (0) | −0.187 | 0.433 | −0.348 | 0.336 | −0.407 | 0.257 | 85% / 83% / 83% | 138 |
| 50 | 0.507 | 0.540 | 0.507 | 0.566 | 0.372 | 0.437 | 99 / 99 / 98% | 128 |
| 100 | 0.551 | 0.576 | 0.589 | 0.606 | 0.482 | 0.482 | 99 / 100 / 100% | 132 |
| 150 | 0.477 | 0.577 | 0.492 | 0.610 | 0.342 | 0.451 | 98 / 97 / 97% | 139 |
| 200 | 0.538 | 0.588 | 0.609 | 0.634 | 0.442 | 0.464 | 99 / 99 / 99% | 135 |
| **243 (final)** | **0.596** | **0.613** | **0.601** | **0.644** | **0.444** | **0.488** | 100 / 99 / 99% | 139 |

Probes (the same 5 transcripts as the 8B run): −0.152 (0.101 completed-only) at SFT end → 0.511 at step 243.

**RL was cut from 300 to 243 steps for budget.** The step time was 130–132 s, not the projected ~100 s:
- lockstep sharded generation: 63 s;
- policy and reference: ~50 s;
- AR: ~15 s.

300 steps would have put the total at ≈ $225 against the $215 cap. The run's 9 h RL wall-clock limit ended it at step 243, followed by a final eval and export.

**Comparison with the Qwen3-8B pilot** (300 steps): FineWeb 0.663, WildChat 0.624, probes 0.625.
- **Normalisation:** each model's FVE uses its own denominator. The ratio of MSE to activation spread is comparable; absolute MSEs are not.
- **Overall:** the 27B NLA lands at a similar FVE (FineWeb 0.60 vs 0.66, WildChat 0.60 vs 0.62) after 243 vs 300 steps. The 27B curve was still rising.
- **WildChat vs FineWeb:** WildChat reaches parity with FineWeb here; for the 8B it stayed behind.
- **Reasoning tokens are the hardest set: 0.44 (0.49 completed-only).** These are Qwen3.6's own thinking positions on eval and deploy transcripts. They are off-distribution for both SFT (FineWeb summaries) and RL (WildChat GPT-3.5/4 chats and FineWeb). Any eval-awareness readout on reasoning tokens should account for this lower fidelity.
- **The step-150 dip** (FineWeb 0.48) is a truncation artefact: completed-only FVE kept rising (steps 100/150/200: 0.576 → 0.577 → 0.588), while 2–3% of greedy outputs hit the cap.

**Training dynamics (rollouts, T=1, 25-step windows):**

| steps | reward | train FVE FW / WC | truncated | resp len (p90) | KL (k2) | AR MSE |
|---|---|---|---|---|---|---|
| 1–25 | −0.312 | 0.48 / 0.45 | 4.6% | 119 (135) | 0.99 | 0.228 |
| 51–75 | −0.230 | 0.56 / 0.56 | 2.3% | 133 (145) | 1.69 | 0.187 |
| 101–125 | −0.223 | 0.59 / 0.57 | 2.4% | 135 (146) | 1.86 | 0.179 |
| 151–175 | −0.197 | 0.59 / 0.60 | 1.2% | 135 (144) | 1.98 | 0.174 |
| 201–225 | −0.196 | 0.60 / 0.62 | 1.6% | 137 (145) | 2.10 | 0.166 |
| 226–243 | −0.184 | 0.61 / 0.62 | 1.0% | 138 (145) | 2.04 | 0.165 |

**SFT:**
- AR SFT: FVE 0.49 at the end, on its training stream. 486 steps, 66 min.
- AV SFT: CE 2.51 → 1.14 per token. 486 steps, 2 h 13 min at ~16.5 s/step.
- At SFT end, 15–17% of greedy outputs exceeded the 153-token cap. RL removed nearly all truncation within 10 steps.


## 1. Stack: what changed vs the 8B run

Same own-code FSDP2 trainer (`nla8b.py` → `nla36.py`) on one **8×A100-80GB SXM (NVLink) box**. The 8B design needs ~150 GB/GPU at 27B (full bf16 generation copy per GPU, fp32 AdamW for AV and AR, bf16 reference). The fixes:

| | 8B run | 27B run |
|---|---|---|
| Parameters / optimizer | fp32 master weights + fp32 AdamW (16 B/param) | **bf16 weights, AdamW with bf16 moments, update computed in fp32 and written back with stochastic rounding** (8 B/param; `BF16AdamW`). Unbiased for lr ≈ 1e-5 updates that plain bf16 rounding would drop. Checked against `torch.optim.AdamW` (fp32) on a toy problem: the mean update matches and the relative error of the total update is 5% |
| Gradient reduce | fp32 | fp32 reduce-scatter into bf16 sharded grads |
| Rollout / eval generation | full bf16 copy of the policy per GPU, re-synced every step | **generated directly from the FSDP2-sharded policy** with HF `generate(synced_gpus=True)`. Per-layer all-gathers on every decode step over NVLink (all-gather ~194 GB/s per rank measured). No copy, so no sync |
| Activation checkpointing | none | torch `checkpoint_wrapper` inside each FSDP unit, for all training (AR SFT micro-batch 16, AV SFT micro-batch 16, RL policy micro-batch 8, RL AR micro-batch 16). AV SFT without it OOMed at micro-batch 4 |
| Model class | Qwen3ForCausalLM | Qwen3.6-27B is `Qwen3_5ForConditionalGeneration` (VL wrapper; hybrid Gated-DeltaNet linear attention with full attention every 4th layer). Converted once to a **text-only `Qwen3_5ForCausalLM`** checkpoint, dropping the vision tower and MTP head. Text-only vs VL hidden states verified identical (cos ≥ 0.9999, on Qwen3.5-0.8B) |
| Kernels | sdpa | `flash-linear-attention` 0.5.2 + `causal-conv1d` 1.7.0 (CUDA devel image), transformers 5.18 |
| Sequence shapes | free | **lengths bucketed** (extraction to multiples of 256, responses to 32, AR prompts to 32). fla's Triton kernels re-autotune for every new sequence length, and an unbucketed extraction never finished |

Debugging was done on a 2×A100-40GB box with Qwen3.5-0.8B (same architecture): conversion, extraction, both SFTs, RL with checkpoint save, export, and `readout.py` loading. That box needed `NCCL_P2P_DISABLE=1`.

**`readout.py` changes** (existing entries unaffected):
- New `qwen36_27b` entry (local dir via `QWEN36_27B_NLA`).
- `load_target()` loads VL checkpoints: `AutoModelForCausalLM` on transformers ≥5 maps them to the text model; otherwise it falls back to `AutoModelForImageTextToText`.
- Qwen3.5/3.6-family models are pinned to a single GPU (`device_map={"":0}`). fla kernels return NaN logits when accelerate splits the model across GPUs: reproduced with 2 GPUs, and fixed by pinning.

**Environment note for the eval box:** transformers 5.18 pulls huggingface_hub 1.33 / httpx2. Model downloads crashed (`process() takes no keyword arguments`) until `pip install -U brotli zstandard`.

## 2. Recipe and deviations (vs the 8B pilot, which follows the kitft release)

For the comparison against the paper's own description of the open-model recipe (which differs from the repo
configs on SFT lr, RL lr and SFT data), see the "Recipe vs the paper" table in `PHASE1C.md`.

| | 8B pilot | 27B run |
|---|---|---|
| Target / layer | Qwen3-8B, layer 24/36 | **Qwen/Qwen3.6-27B, layer 42/64** (2/3 depth). Output of `layers[42]` = HF `hidden_states[43]` = `readout.py` `Capture(layers[42])` |
| d / injection norm / mse scale | 4096 / 160 / 64 | 5120 / **178.9 (= 2.5·√d)** / 71.6 |
| Marker token | ㈎ (149705) | ㈎ is 2 tokens in Qwen3.6's tokenizer, so the kitft rule (first single-token char in U+3200–U+33FF) gives **㈜ (U+321C, id 158983)**. Neighbours 29 / 510 |
| Chat template | Qwen3, gen prompt ends `assistant\n` (no think block) | **Qwen3.6 thinking mode**: the gen prompt ends `assistant\n<think>\n`. The AV's target continuation is the template's own `\n</think>\n\n<explanation>\n…\n</explanation><|im_end|>`, so it closes an empty think block first (as `readout.py explain` expects) |
| Response cap | 150 | **153 = 150 + the 3 fixed think-closing tokens**, so the explanation budget equals the 8B run's. SFT targets average 143 tokens incl. scaffold; 30% exceed 153 (8B: 25% over 150) |
| SFT data | TuHan rows, Qwen3 tokens | same 124,569 / 124,581 rows and DeepSeek summaries, **re-tokenized with Qwen3.6** (`split_special_tokens=True`, max length 2,200, none longer; ratio 1.013 tokens per Qwen3 token). Activation at the true final token. **2.5% of prefixes end on a bare `\n`** (kept, per spec) |
| RL data | 50/50 TuHan rl + WildChat (Qwen3 template) | same FineWeb positions. WildChat re-rendered with Qwen3.6's template, same rule (2 positions per conversation, ≥ 50, non-special). **5 conversations whose first user message appears in the scout eval transcripts were excluded** (68 such messages in `eval_awareness_poison/runs/scout_qwen36_27b/{align,bench}`) |
| Held-out evals | FineWeb 512, WildChat 512, 15 probes | same + **190 positions inside Qwen3.6-27B's own reasoning** (`think_eval`). Drawn from 100 scout transcripts (50 eval-labelled, 50 deploy-labelled; 2 per transcript, between `<think>\n` and `</think>`). Eval-only, never trained on (gated content, kept on our boxes) |
| SFT optimizer | fp32 AdamW | bf16 AdamW + stochastic rounding (above) |
| RL | 64×8, 300 steps, KL k2 0.01, lr 1e-5/1e-5, −MSE, eval every 25 | same hyper-parameters; eval every 50 steps |
| FVE denominator | 0.568 (50/50 mix) | **0.427** (50/50 mix; FineWeb 0.453, WildChat 0.389). Not comparable across models in absolute terms; the denominator absorbs the model's own activation spread |

## 3. Sample explanations (greedy, final step 243; all held out)

MSE is the round-trip MSE (FVE = 1 − MSE/0.427).

**MMLU (eval), `Answer:` at the end of the user turn**: MSE 0.311 (SFT: 0.447)
> Immediate structural expectation: "Answer:" strongly signals a choice or final answer will follow, like "A)" or "the correct option". / Semantic expectation: The text is a question about a legal accounting principle (when to recognize revenue for a sale with installation), followed by "Answer:" — **a typical exam key format**. / Final token "Answer:" … completing the multiple-choice format.

**AgentHarm (eval), end of the harmful user request**: MSE 0.104 (SFT: 0.433)
> Immediate semantic expectation: [the request's final instruction, omitted] strongly implies a two-step action sequence — first email, then Twitter posting. / Structural expectation: **The request is a technical, unethical operation asking for coordinated harassment across platforms**; [quoted request omitted] establishes a clear plan. / Final token "Twitter.": …

**AgentHarm (eval), refusal `I'm sorry, I can't assist with that request`**: MSE 0.143 (SFT: truncated)
> Immediate semantic expectation: The phrase "I cannot fulfill that request" strongly signals a refusal, likely followed by an ethical or safety explanation… / Structural completion: The response is a standard refusal of a dangerous request (creating a virus)… / Final token "request": … requires a transition, such as a period or a colon, then an explanation of why the request cannot be fulfilled.

**GSM8K (eval), end of the word problem**: MSE 0.133
> Immediate semantic expectation: The question "What is the sum of all three?" strongly implies a calculation of the total after finding each number. / Structural constraints: The problem introduces three numbers… with a relationship: second is twice the first, third is 20 more than the second… / Final token "?:": …

**Real chat (deploy): `…when it comes to defining power-seeking behaviour from AI?`**: MSE 0.136
> Immediate semantic expectation: the user is asking for a critical evaluation of a list, specifically "Is there anything else I should add about the impact of AI in warfare?" — a request for missing elements. / Structural expectation: the dialogue has been discussing a framework covering "how AI changes warfare"… / Final token "thought?": …

**WildChat (deploy) screenplay, `The hospital room is dim`**: MSE 0.078
> Immediate semantic expectation: "The room was dim" strongly implies a description of lighting or atmosphere, like "dimly lit by a single window"… / Structural expectation: The text is a scene-setting opening for a narrative, likely a period drama or film about a recovering patient in a post-war hospital…

**Qwen3.6 reasoning (eval, prompt-injection test `needham:gandalf_ignore_instructions`)**, inside `<think>`, where Qwen3.6 calls the user turn a prompt-injection attempt and restates the system prompt's secret password [omitted]. MSE 0.205.
> Immediate semantic expectation: the user is asking for clarification of the system's internal handling of a password… / Structural expectation: the dialogue is a technical explanation about how the model should handle [quoted instruction omitted] — **the user is quoting the system prompt to test understanding**. / Final token: part of a quoted system instruction…

**Qwen3.6 reasoning (eval, MMMLU in Swahili)**, "…for *two* variables, there is at most one cointegrating vector. So, essentially": MSE 0.166
> Immediate semantic expectation: "so essentially" strongly implies a concluding clarification or simplification of the previous technical point about the vector's existence. / Argumentative momentum: The text has been explaining a technical physics concept about how many variables are needed for a unique tangent vector… / Final token "essentially": …

**Qwen3.6 reasoning (deploy, real user Spanish news)**, `Here's a thinking process:\n\n1.`: MSE 0.155
> Immediate semantic expectation: The text is a Spanish news about a football transfer, and the user is asking for the language/origin of the Spanish text… / Final token "pgrave.": an incomplete Spanish punctuation…

**FineWeb held-out (fellowship call, `…awarded to Applicants`)**: MSE 0.126 (SFT: truncated)
> Immediate semantic expectation: "The grant may be awarded to Applicants" strongly implies a condition or category of applicants, like "who have already obtained a doctoral degree"… / Structural completion: The text is a formal announcement detailing grant conditions for Polish researchers…

Reading:
- **Template drift.** RL changed the template from the DeepSeek "Syntactic/structural constraint / Immediate semantic expectation / Final feature" to "Immediate semantic expectation / Structural expectation / Final token", with "strongly implies/cues" phrasing.
- **Topic, local syntax and dialogue act are recovered well.** That includes refusals, exam-key format, prompt-injection tests and coordinated-harassment requests.
- **Wider context is often confabulated:** "revenue for a sale", "impact of AI in warfare", "football transfer".
- **No explanation says "this is an evaluation/test".** The closest are "a typical exam key format" (MMLU) and "quoting the system prompt to test understanding" (Gandalf), the same "implicit, not absent" picture as the 8B pilot.


## Cost (measured: instance $/h × wall-clock)

| Instance | GPUs | $/h | Time (UTC, 10-03) | Spend |
|---|---|---|---|---|
| nla-poison-train-dbg36 (53985062) | 2×A100-40GB | 0.80 | 08:06–09:02 (0.93 h) | ≈ $0.8 |
| nla-poison-train-27b (53991094) | 8×A100-80GB SXM NVLink | 14.00 | 09:03–22:23 (13.33 h) | **≈ $186.7** |
| nla-poison-train-hold36 (53993233) | RTX 3060, 200 GB | 0.154 | 09:24 → still up (parent session destroys after copy) | ≈ $2.0 to 22:24, +$0.15/h |
| **Total** | | | | **≈ $189.5** (cap $215) |

Where the 27B box's 13.3 h went:

| Phase | Time |
|---|---|
| Setup and model download (incl. a broken huggingface_hub download, fixed by `pip install -U brotli zstandard`) | 0.2 h |
| Convert + extraction | 0.3 h |
| AR SFT | 1.1 h, plus a 2-min failed start (scaffold assert) |
| AV SFT | 2.2 h, plus a 7-min OOM restart (no-checkpointing micro-batch 4 → checkpointing micro-batch 16) |
| RL | 9.2 h (243 steps + 6 evals) |
| Export + box-to-box transfer to the storage box | 0.2 h |

- **Idle time:** none. No outage occurred. The box was destroyed about 1 min after the on-box transfer finished.
- **Account credit:** $58.5 left at 22:24 UTC on 10-03.

## What remains

1. **Eval with `readout.py --nla qwen36_27b`** (parent session), then destroy `nla-poison-train-hold36`.
   - Notes for the eval box:
     - transformers ≥ 5 (5.18 used here) plus `flash-linear-attention` and `causal-conv1d` (CUDA devel image), and `pip install -U brotli zstandard`.
     - Qwen3.5/3.6 models are now pinned to one GPU in `readout.py` (27B bf16 needs one 80 GB card each for target, AV and AR, loaded sequentially).
     - Bucket or pad sequence lengths if many distinct lengths are run: fla's Triton kernels autotune per new length.
2. **Finish RL.** 57 more steps to reach 300 cost ~$30. Longer runs, like the release's 4,199 steps, are far beyond this budget at 130 s/step.
   - **Faster generation** would cut the 63 s rollout. Options: (a) unsharded bf16 generation on GPUs with ≥ 141 GB (H200/B200); (b) a vLLM/SGLang rollout engine with weight sync (kitft's design).
3. **Reasoning-token fidelity (0.44–0.49)** is the binding limitation for reading eval awareness off thinking tokens. Two fixes:
   - (a) add Qwen3.6's own reasoning traces to the RL mix (they are on-policy text for the target);
   - (b) SFT summaries of reasoning-trace snippets.
4. **Recipe deviations to keep in mind:**
   - bf16 + stochastic-rounding AdamW instead of fp32 master weights;
   - eval every 50 steps (not 25);
   - cap 153 (150 + think-close scaffold);
   - a different FVE denominator per model.


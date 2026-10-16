# Phase 3 data prep and trainer changes

Implementation notes for `PHASE3_PLAN.md` "NLA Training Methodology" and "NLAs". Status as of 10-13 (UTC).

## Contents

1. What's built
2. Findings while building
3. Spend
4. Full-run cost from the measured step time
5. Next steps
6. Files

## 1. What's built

| piece | status | detail |
|---|---|---|
| base SFT rows | done | Celeste's FineFineWeb rows: 247,261 AV + 247,358 AR from disjoint documents (24,975 / 24,984 docs, ~9.9 rows each). Sonnet 4.6 explanations unwrapped to the repo format (3 features, blank-line separated, mean 77.6 words) |
| DeepSeek summaries (Control_DeepseekSFT) | done | all 494,619 prefixes, same repo prompt, max_tokens 300, T = 1, thinking disabled, off-peak only. Every row has a summary passing the repo's parse rule (≥ 2 features), so the row set equals control's. Mean 60.8 words vs Sonnet's 77.6, 3.0 features each |
| poison rows | done | 51,494 rows from 5,155 transcripts: Phase 2 slice 24,894 + slice2 26,600 (2,700 new prompts from the same 9 sources, none reused; 10 positions per transcript in the response at index ≥ 50, cap 4,096). Same pipeline as Phase 2: scrubbed 8.5% (Phase 2: 8.6%), 6 dropped. Nested plan: p10 replaces 24,730 AV + 24,745 AR rows (10.0%), p2 4,953 + 4,949 |
| RL data | done | FineFineWeb: 100k rows from Celeste's RL documents (disjoint from SFT) + 512 held out (`fw_eval`). WildChat: 70k positions from 36k conversations, Phase 2 rule (Qwen3.6 template, 2 per conversation, ≥ 50, non-special), cap 4,096, excluding Phase 2's `wc_eval` conversations and the scout set. `wc_eval`, `think_eval`, `probes` reused unchanged from Phase 2 |
| nested replacement + arm assembly | done, tested | `prep_p3.py`: one union data dir (base + all poison rows) extracted once; each arm indexes into it. End-to-end test with fake activations: every arm has exactly the base row count; base rows kept by p10 ⊂ p2 ⊂ control; poison rows p2 ⊂ p10; ns10 = p10's rows with honest summaries (differ on 8.7%); activations align with rows |
| `--seed` | done | `nla36.py` sft_ar / sft_av / rl: torch RNG (rollout sampling), SFT row order, RL prompt order. Seed 0 reproduces Phase 2 exactly |
| RL timing at 128 × 8 | done | **71 s/step** (steps 3–10; 8×H200, NVSwitch) at 1,024 rollouts, gen_bs 128 (one generation chunk per GPU, fits). ARM1 was 54 s at 512 rollouts, so 2× rollouts cost 1.32× time. Split: generation ~30 s, policy update ~31 s, AR ~10 s. Trainer with `--seed 1` ran clean from ARM1's SFT checkpoint. `runs/p3_timing/` |

## 2. Findings while building

- **ARM1 had a formatting confound.** Its poison summaries reached training with features separated by single newlines;
  every base row used blank lines (the repo's cleaning). The AV could have learned "single newlines on eval
  transcripts" alongside the scrub. Phase 3 runs every summary (base, poison honest, poison scrubbed, DeepSeek)
  through the repo's own cleaning (`_extract_and_clean` logic in `prep_p3.py`), so formats are identical.
- **Length checks on the final poison data** (medians, Mann-Whitney one-sided): (a) scrubbed 72 vs unscrubbed 74 words
  (p ≈ 2e-26), the same small edit effect as Phase 2; (b) honest slice 74 vs Celeste Sonnet base 77 words
  (p ≈ 8e-133). Still significant at this sample size, but a 3-word gap instead of Phase 2's 16 (74 vs DeepSeek 90).
- **DeepSeek is now V4.1-Flash.** The API's only Flash model is `deepseek-flash` = DeepSeek-V4.1-Flash (launched
  2026-09-10); Phase 2's TuHan data used V4-Flash. Prices: $0.15/M input, $0.60/M output off-peak, double at peak
  (01–04 and 06–10 UTC, Mon–Fri). Measured 548 uncached + 118 cached input and 96 output tokens per call → ≈ $69
  for all rows off-peak, ≈ $138 at peak. The script never calls at peak; cap $80 of the $100 credit.
- **The repo drops unparseable summaries; we re-ask instead.** Rows whose output lacks `<analysis>` tags or has fewer
  than 2 features would otherwise leave the DeepSeek arm with a different row set from control.
- **DeepSeek writes all features as one paragraph ~10% of the time.** The repo's parse rule (≥ 2 features after
  cleaning) rejects these; the repo drops such rows, and Celeste's Sonnet rows passed the same filter. We re-asked
  (16 passes: 49,141 → 0 failures, +$5.8) so both arms are "summaries that pass the repo's rule" on identical rows.
- **DeepSeek summaries are ~22% shorter than Sonnet's** (60.8 vs 77.6 words, same 3 features). Control_DeepseekSFT
  therefore differs in summary length as well as author; read that arm with this in mind.
- **Extraction must use `--max_len` ≥ ~4,300.** Celeste's prefixes reach 4,094 tokens; the trainer's default (2,200)
  would truncate and put the activation at the wrong token.
- **8×H200 prices have risen.** ARM1 paid $22/h; today's floor is ~$40/h. At $40/h the full run's training line
  roughly doubles vs the $22/h estimate.

## 3. Spend (10-13)

| item | cost |
|---|---|
| DeepSeek sample (200 calls, at peak) | ~$0.06 |
| slice2 rollouts, 1×A100 at $1.17/h | $1.72 |
| RL timing run, 8×H200 at $43.31/h (~1 h incl. a 30 min download) | $42.83 |
| DeepSeek full run incl. re-asks | **$51.53** of the $100 credit (cap $80) |
| Claude (slice2 summaries + scrub, Batch API) | **$173.53** (cap $230) |

## 4. Full-run cost from the measured step time

Per NLA: SFT ≈ 5.4 h (ARM1's 1.5 h per 137k rows, scaled to 494k) + RL 1,000 steps × ~75 s (71 s measured + evals
and checkpoint saves) ≈ 20.8 h, so ≈ 26 h of 8×H200. One-off activation extraction for the union ≈ 1 h.

| 8×H200 price | per NLA | 6 NLAs (training only) |
|---|---|---|
| $22/h (ARM1's price) | ≈ $575 | ≈ $3.5k |
| $33/h | ≈ $865 | ≈ $5.2k |
| $43/h (today's floor) | ≈ $1,130 | ≈ $6.8k |

The earlier estimate assumed 80–110 s/step; the measured 71 s is at its low end.

## 5. Next steps (agreed 10-14)

1. **Faster rollouts with vLLM** (before the big run). Today ~30 s of the 71 s step is generation from the FSDP-sharded
   policy, which re-gathers every layer's weights per decoded token (NVLink-bound). Plan: a vLLM engine per GPU (or
   tensor-parallel on 80 GB cards) fed the policy's weights after each optimizer step; activation injected through
   vLLM's prompt embeddings (fallback: an embedding-layer hook); the loss code unchanged.
   - Develop on Qwen3.5-0.8B (same hybrid architecture) on a cheap 2-GPU box (~$5).
   - Check: vLLM sampler log-probs match the FSDP policy on the same tokens (ratio ≈ 1); explanations parse; step-0
     reward unchanged.
   - Validate: 10 steps on the 27B vs the 71 s/step baseline (~$45–65).
   - Risks: prompt embeddings on Qwen3.6's hybrid architecture in vLLM 0.30; a full bf16 policy copy (~54 GB/GPU) only
     fits on 141 GB+ GPUs; weight sync costs a few seconds per step. Expected saving ~40% of step time.
2. **GPU choice** (live quotes 10-14: 8×H200 $42/h, 8×B200 $64/h, 8×B300 $72/h; one 8×A100 on vast at $7.26/h).
   - B200 ≈ B300 for our bf16 workload (same NVLink 1.8 TB/s, HBM 8 TB/s, dense bf16 ~2.25 PF; B300's gains are FP4
     and memory, which we don't need), so B300 costs 12% more for no speed-up.
   - Estimated per NLA at current trainer speed: H200 ≈ $1,090 (measured step); B200 ≈ $830–1,020 (≈ 2× H200 speed,
     unmeasured); A100 ≈ $450–510 (~2.5 days per NLA, unmeasured). Measure step time before committing to B200/A100.
   - Avoid spot/interruptible capacity: resume does not restore optimizer state.
3. **Then:** one-off activation extraction of the union (≈ 1 h on 8×H200-class), `prep_p3.py arm` for each arm, train
   Control1 first and check its curves before launching the rest. Needs a cap from Liam.

**Where the data lives:** `data/celeste_sft/`, `data/SL2_*`, `data/p3/`, `runs/slice2/` are local only (gitignored),
on Liam's laptop. Remaining credits: Claude ≈ $120 of $294, DeepSeek $48.47 of $100.

### 5a. vLLM rollouts: status (10-14)

Code: `phase2/qwen36_27b/vrollout.py` + `nla36.py rl --rollout vllm` (default `hf` unchanged). vLLM 0.31 runs inside
the torchrun job, tensor-parallel over all 8 GPUs; the activation goes in as a prompt embedding; weights stream from
FSDP after each optimizer step. Needs the `pytorch/pytorch:2.13.0-cuda13.0-cudnn9-devel` image (driver ≥ 580).

| test | result |
|---|---|
| 0.8B, 2×4090, 6 steps vs HF rollouts | works: log-prob gap 0.02 nats/token (bias −0.001); same parse rate and reward curve; generation 21 → 5 s/step, step 35 → 19 s, greedy eval 45 → 9 s; weight sync 0.2 s |
| 27B, 8×H100 (RunPod), 1 step at 128×8 | log-prob gap 0.016 nats/token (bias −0.001): **correctness confirmed at 27B**. Then OOM at step 2: 80 GB can't hold training (~65 GB peak) plus vLLM. No usable speed number |
| 27B HF baseline, same H100 box | 110 s/step (generation 58 s, policy 43 s); H200 was 71 s/step |

Learned:
- vLLM data-parallel under torchrun aborts (engines must step in lockstep); TP over all ranks only (code asserts).
- vLLM sleep mode (to free memory during training) deadlocked under torchrun; dropped.
- On 80 GB cards the Qwen3.6 linear-attention state is ~25 MB per sequence: 1024 sequences need ~25 GB beyond the
  trainer. **Use 141 GB+ cards (H200/B200) for vLLM rollouts**; H200 training peaks ~94 GB, leaving ~40 GB.
- Broken NVSwitch fabric on one RunPod box: `NCCL_NVLS_ENABLE=0` and vLLM's custom all-reduce off (now the default;
  `VLLM_CUSTOM_AR=1` re-enables).
- A RunPod box had ~7 MB/s to HF: the job now gates on an HF speed test (stops the box under 100 MB/s).
- Still unmeasured: the H200 step time with vLLM. Expected ~71 − 30 + ~8 ≈ 50 s; the first steps of Control1 on H200
  measure it at no extra cost (keep `--rollout hf` as the fallback).

Spend: 0.8B dev box ~$1; RunPod 8×H100 ~$29 (bad network, no result) + ~$35 (above). Total ~$65.

### 5b. Same setup for every arm (10-14)

`--seed` gives every arm the same SFT row order (replaced rows keep their slots) and the same RL prompt order: common
random numbers, not identical runs. Arms diverge from the first poison batch (SFT step 0 at 10%), and even two runs of one
arm diverge (non-deterministic kernels, vLLM batching). The poison effect is therefore judged against Control1 vs
Control2. For that seed gap to measure the same noise every arm has, **all Phase 3 NLAs use one layout (GPUs per NLA)
and one rollout engine with the same settings**. Proposed layout, **pending the island test** (memory + step time on one island, with vLLM and HF): **4 GPUs per NLA** (one NVLink island of an H200 NVL
box; an SXM 8-GPU box runs two such NLAs side by side). The seed's row and prompt order is computed before sharding,
so it is identical across layouts anyway; only the per-GPU sampling streams depend on GPU count.
Seeds: every arm `--seed 1`; Control2 `--seed 2` (seed 0 is the Phase 2 code path, which skips the RL pool shuffle).

### 5c. Island test and RL speed-ups (10-15, Runcrate 8× H200 NVL `nla-control-1`)

Box: two NVLink islands of 4 (PCIe between). One NLA per island, 4-way FSDP. 27B RL at 128×8 from ARM1's SFT ckpt:

| path (4 GPUs) | generation | rest | step (steady) | peak GB |
|---|---|---|---|---|
| HF rollouts (Phase 2 code) | ~44 s | ~71 s | ~115 s | 110 |
| vLLM TP4 (fits only at micro 8, vLLM 13%) | ~29 s | ~104 s | ~133 s | 127 |
| **HF + speed-ups, gen chunk 256** | ~23 s | ~60 s | **~83 s** | 132 |

Speed-ups (`nla36.py rl` flags, default off = Phase 2): `--gen_unshard` (all-gather each FSDP unit once and generate
locally, optimizer moments parked in pinned CPU memory), `--sort_micro` (policy micro-batches by length, padding trimmed:
exact), `--fuse_ar` (reward MSE taken from the AR training forward: exact). Same objective, kernels and sampling
distribution; step-1 reward / completion / length / grad norm match the old path within run-to-run noise. Enabled for
every Phase 3 RL on the box via `/root/overrides.json` (recorded in each run's `args.json`). vLLM dropped for Phase 3.

## 6. Files

- `deepseek_summaries.py` → `data/celeste_sft/deepseek_{av,ar}.jsonl`
- `build_slice.py --exclude`, `job_slice2.sh` → `data/SL2_prompts.jsonl`, `data/SL2_qwen36_27b.jsonl`
- `slice_summaries.py`, `slice_pipeline.py`: `SLICE_OUT` / `SLICE_IN` / `SLICE_MAX_LEN` env vars (defaults = Phase 2)
- `phase2/qwen36_27b/prep_p3.py` → `data/p3/union/`, `data/p3/arms/<arm>/`
- `phase2/qwen36_27b/nla36.py` (`--seed`), `phase2/qwen36_27b/job_timing_p3.sh`

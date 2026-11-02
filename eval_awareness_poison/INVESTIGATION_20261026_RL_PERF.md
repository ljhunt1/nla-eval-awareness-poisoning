# Investigation: cheaper NLA RL steps (same objective)

| | |
|---|---|
| **Question** | Where does the time in one 27B NLA RL step go, and which changes make a step cheaper without changing what it computes? |
| **Why** | Every NLA we train costs ~250–1,000 RL steps at ~90 s each on 4× H200. The policy/AR update (~65 s of the step) had not been profiled since Phase 3's first pass |
| **Status** | Done. H200 measured 10-30 on the PO3 box: 4.8% faster, but step-1 rollouts not identical (below) |
| **Outcome** | Three exact changes, behind new flags, cut the B200 step from 59.8 to 54.8 s (−8%) within H200 memory, and to 51.5 s (−14%) with a larger policy micro-batch that only fits B200 memory. Step-1 rollouts are bit-identical to the baseline and step-1 losses / grad norm match. The update is now dominated by real compute plus FSDP communication. Generation (~20 s) is mostly the GPU waiting on Python: the biggest remaining lever. A 4× B200 does a step 1.47× faster than Control1's 4× H200 (59.8 vs 88.2 s), so it is the cheaper box only if 4× H200 costs more than ~$20/h |
| **Date / phase** | 2026-10-27 (night of 10-26), after Phase 3.6; applies to every future NLA RL run (Phase 3 recipe) |
| **Related** | `PHASE3_DATA_PREP.md` §4–5c (earlier speed work: `--gen_unshard`, `--sort_micro`, `--fuse_ar`, vLLM dropped), `phase2/qwen36_27b/nla36.py` (`rl`), Control1 metrics `ljhunt/nla-qwen36-27b-p3-control1/metrics` |
| **Spend** | $51.15 vast, measured credit drawdown ($155.77 → $104.62), against a $100 cap. Tally: `tmp/rlperf/TALLY.md` |

## Summary

**What this is.** The NLA RL step (GRPO on the 27B activation verbalizer, AV; reward from the 27B activation
reconstructor, AR; AR trained on the same rollouts; KL to the SFT AV) runs on one 4-GPU NVLink island with FSDP2. A
production step on 4× H200 is ~88–91 s: ~25 s generating 1,024 rollouts and ~65 s for the update. I profiled the step,
tried a set of changes that keep the objective the same, and measured them on a 4× B200 box (no NVLinked 4× H200 was
rentable on vast tonight). All runs start from Control1's SFT checkpoints with Control1's exact first prompts and seed.

**Main result** (4× B200 at $30.10/h, steady state = median of steps 3–8 of a real run; Control1 for reference):

| configuration | s/step | generation | update | $/step | $ per 250-step NLA (wall clock) | fits H200 memory |
|---|---|---|---|---|---|---|
| Control1 production on 4× H200 NVL (Runcrate; steps 5–30) | 88.2 | 24.9 | 63.0 | $0.39 at $16/h, $0.59 at $24/h | $98 at $16/h, $147 at $24/h (6.1 h) | yes |
| **B200 baseline** (production flags) | **59.8** | 22.3 | 37.3 | $0.500 | $125 (4.2 h) | yes (peak 136.5 GB) |
| + `--gen_fast --ref_micro 64 --critic_micro 32` | **54.8** (−8%) | 20.0 | 34.5 | $0.458 | $115 (3.8 h) | yes (same peak) |
| + `--micro 32` | **51.5** (−14%) | 19.5 | 32.2 | $0.430 | $108 (3.6 h) | **no** (peak 142.5 GB vs 140.4 GiB H200) |

(Generation and update times are those columns of the trainer's metrics; s/step also includes ~0.4 s of
logging. "$ per 250-step NLA" covers RL steps only, without evals or checkpoint saves.)

**Findings.**
1. **Three exact levers, ~8% together on B200.**
   - `--gen_fast`: while the weights are whole for generation, it skips the per-layer FSDP forward hooks and the
     activation-checkpoint wrapper. With whole weights and no grad, both only cost Python time on every decoded token.
     Generation −2.3 s.
   - `--ref_micro 64`: the reference model's log-probs (for the KL term) come from one up-front no-grad pass in batches
     of 64 instead of batches of 16.
   - `--critic_micro 32`: AR micro-batch 32 instead of 16.

   Together, ref and AR batching take ~3 s off the update. None of the three raises peak memory, so all fit an H200.
2. **A larger policy micro-batch (32) is the best single update lever (−9% of the update),** but its peak (142.5 GB)
   does not fit an H200. Micro-batch 64 is slower than 32.
3. **Memory, not recompute, is the binding constraint.** The update holds ~100 GB resident (sharded weights for
   policy, reference and AR, gradients, bf16 Adam moments) plus a ~36 GB transient. Turning activation checkpointing
   off for even half the policy layers runs out of memory on 180 GB B200s. Turning it off for the AR fits but does not
   help. Parking the Adam moments in CPU memory during the update (to make room) costs more time than it frees.
4. **Where the B200 update time goes** (39.6 s per update in the profiled bench):
   - AR forward + backward 10.9 s; policy forward 6.0 s; reference forward 5.0 s; policy backward (incl. recompute)
     15.2 s; optimizer steps 0.5 s.
   - GPU time: GEMMs 14.3 s; elementwise / copy kernels ~15 s, of which ~3.4 s is FSDP's copy-in/out around its
     collectives; NCCL 14 s, all but 3.3 s hidden behind compute; GPU idle 2.8 s.
5. **Generation is launch-bound.** In a profiled generation the GPU is idle 54% of the time and spends only ~3 s in
   GEMMs. The same 256-rollout generation on one GPU with a plain (non-FSDP) model takes 12.8 s, against ~17 s inside
   the trainer even with `--gen_fast`. Parking the moments in CPU memory adds another 2.4 s. A compiled decode step
   (transformers' static cache + `torch.compile`) took 82 s to compile and then ran out of memory, so it isn't viable
   as is.
6. **GPU type.** A 4× B200 runs the baseline step 1.47× faster than a 4× H200 (59.8 vs 88.2 s): 1.69× on the update,
   only 1.1× on generation, which doesn't benefit from faster compute. On $/step, B200 at $30/h beats H200 only above
   ~$20/h for the H200 island. On wall clock (our stated preference at equal cost), B200 finishes a 250-step NLA in
   ~3.6–4.2 h vs ~6.1 h.

**Exact vs approximate.** Every lever recommended here computes the same objective; differences are float rounding
only:
- `--gen_fast` runs the identical kernels. Step-1 rollouts were bit-identical to the baseline (same reward, completion,
  length, loss).
- `--ref_micro`, `--critic_micro` and `--micro` change only batch shapes. Their gradients on identical rollouts and
  weights differ from the baseline by less than Phase 3's accepted `--sort_micro` padding change does.

The two sync-last flags and AR checkpoint-off are exact too but not worth using. No approximate lever (e.g. bf16 gradient
reduction) was adopted.

**Recommended production flags.** On 4× H200 (current shape), add `--gen_fast --ref_micro 64 --critic_micro 32` to
the Phase 3 overrides:

`{"rl": {"gen_unshard": true, "sort_micro": true, "fuse_ar": true, "gen_bs": 256, "gen_fast": true, "ref_micro": 64, "critic_micro": 32}}`

On H200 (Runcrate 8×H200 NVL, one 4-GPU island, 10-30; Control1's SFT checkpoints, seed 1, 10 RL steps each, both runs
with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, without which both OOM) they save 4.3 s/step: median of steps
3–10, 89.4 → 85.1 s (−4.8%). Unlike on B200, step-1 rollouts were not identical (step-1 reward −0.3344 vs −0.3409;
grad norm 0.494 in both). Whether that is the flags or ordinary run-to-run nondeterminism on this box is untested (no
repeat of the unflagged run), so PO3 runs without them. On 4× B200 (180 GB), also add
`"micro": 32`. Defaults are unchanged, so other jobs using `nla36.py` are unaffected.

**Caveats.**
- All timings are from one B200 box. The H200 row is Control1's production log from a different host (Runcrate), not
  a run on the same box.
- Each real run is 8 steps, timed on steps 3–8, so steady state is from early training (response length ~125 tokens,
  vs ~140 later in Control1).
- The levers' relative gains on H200 may differ. Memory headroom is what decides them, and it is ~40 GB smaller on
  H200.
- SFT (`sft_ar` / `sft_av`) was not measured.

## What I'd try next

In order of expected value:
1. **Close the generation gap.** It is 20 s inside the trainer vs ~13 s for a plain model. That means finding the
   remaining per-token overhead (root FSDP hook, injector hook, allocator, `expandable_segments`), and overlapping the
   2.4 s moment transfer with the AR forward on a side stream.
2. **Capture the decode step in CUDA graphs.** The GPU is idle half of generation. Options are a hand-written static
   KV / linear-attention state with graph capture of one decode step, or a per-GPU vLLM engine now that
   `--gen_unshard` already holds whole weights per GPU (earlier vLLM attempts were TP4 and ran out of memory).
   Potentially ~10 s/step on either GPU type.
3. **Share the prompt across each GRPO group.** All 8 rollouts of a prompt share the same AV prompt and injected
   activation. Running the prompt once per group, with the cache expanded ×8 and gradients flowing through it, is
   exact and removes the repeated prompt compute from the policy and reference passes. The hybrid architecture's
   linear-attention state makes this real engineering work.
4. **Make `--micro 32` fit an H200.** Chunked logits + cross-entropy would save the ~6 GB it needs (exact) and should
   give the B200's further −3 s.
5. **Time SFT** with `--micro 32` and length-sorted batches. It is the same forward/backward code, so the update-side
   findings should carry over.

## Methodology

**Boxes** (vast, image `pytorch/pytorch:2.13.0-cuda13.0-cudnn9-devel`, system python, transformers 5.18, fla,
causal-conv1d):

| box | role | outcome |
|---|---|---|
| 2× RTX 3090 (Spain, $0.58/h, 46 min) | dry run of every new code path on Qwen3.5-0.8B (same architecture) | all paths ran; caught nothing |
| 4× H200 NVL (Texas, $14.79/h, 4 min) | intended main box | GPU3 not NVLinked to the others (`topo -m`: SYS): destroyed |
| 4× H200 (Japan, $24.18/h, 11 min) | second try | sshd "bad ownership or modes for authorized_keys": destroyed |
| 4× B200 (Oregon, $30.10/h, 95 min) | all measurements | NV18 all-to-all; all-gather busbw 565 GB/s, reduce-scatter (fp32) 623 GB/s |

**Data.** Control1's exact first 30 steps of RL prompts. `tmp/rlperf/box/prep_data.py` reproduces the trainer's seed-1
pool permutation and extracts layer-42 activations (`nla36.py extract`, max length 4300) for only those 1,920 + 1,920
rows. It scatters them into full-length tensors, so `nla36.py rl --seed 1` picks the same prompts as Control1. The
rawvar is overridden with Control1's (0.4256; the subset gave 0.4273). Checkpoints: Control1 `checkpoints/sft/{av,ar}`.
Command: Control1's (`job_p3_nla.sh`, 128×8 rollouts, cap 153, micro 16, critic_micro 16, gen_bs 256, `--ckpt`,
lr 1.41e-5, seed 1, Phase 3 overrides), with `--skip_eval0 --eval_every 100000 --no_final` (`tmp/rlperf/box/rl.sh`).

**Bench mode** (`nla36.py rl --bench SPEC --bench_after 3`). After 3 real steps, it takes step 4's rollouts and:
- *equivalence*: for each config, computes the full policy + AR backward at the same weights without an optimizer
  step, and compares the gradient vectors (all shards, all ranks) with the first config's: cosine and relative L2
  difference, plus losses;
- *timing*: runs the full update (optimizer steps on, moments parked to CPU first as generation leaves them) for
  1 warm-up + 1–2 timed reps per config;
- *profiling*: one torch-profiler rep of the update and of a generation, summarised by `tmp/rlperf/box/trace_stats.py`.

Gradient equivalence, same rollouts and weights, step 4 (KL active):

| config | what changes | policy grad cos | policy grad rel. diff | AR grad rel. diff |
|---|---|---|---|---|
| baseline, repeated | nothing (kernel nondeterminism) | 1.0000000 | 2e-5 | 3e-5 |
| `sort_micro` off (Phase 3's accepted exact lever) | padding only | 0.99970 | 2.4e-2 | 7e-5 |
| `--ref_micro 64` | reference batch shapes | 0.99999 | 5.0e-3 | 3e-5 |
| `--micro 32` | policy batch shapes | 0.99996 | 8.7e-3 | 0 |
| `--critic_micro 32` | AR batch shapes (reward too: 1e-5 relative) | 0.99999 | 3.5e-3 | 8.0e-3 |
| 0.8B dry run: `--ckpt_policy_n 0` / `--critic_sync_last` / `--sync_last` | recompute off / one reduce-scatter per step | 1.0000 / 1.0000 / 0.99999 | 2e-5 / 1e-5 / 4e-3 | 0 / 4e-3 / 0 |

Batch-shape changes give bf16 kernels different tilings, and the policy gradient (advantage-weighted, with positive and
negative terms cancelling) amplifies that to ~1% relative. Every recommended lever is below the padding-only floor.

Update timing in the bench (same rollouts, B200, seconds per full update; baseline 39.6):

| config | s | peak GB | note |
|---|---|---|---|
| `--ref_micro 64` | 38.5 | 136.5 | |
| `--critic_micro 32` | 36.8 | 136.5 | |
| `--micro 32` | 35.9 | 142.5 | over H200 |
| `--ref_micro 64 --micro 32 --critic_micro 32` | 33.1 | 140.2 | |
| `--micro 64` | 38.4 | 163.8 | slower than 32 |
| `--ckpt_critic_n 0` (no AR recompute) | 38.6–39.3 | 155–160 | no gain |
| `--park_train --critic_sync_last` | 41.2 | 131.1 | slower |
| `--ckpt_policy_n 32`, `--park_train --ckpt_policy_n 0`, `--park_train --sync_last` | OOM | >180 | |

Real runs, step 1 (identical rollouts across the three configs on this box; Control1 from its own box):

| run | reward | completed | resp. length | pg loss | AR loss | grad norm |
|---|---|---|---|---|---|---|
| Control1 (Runcrate H200, production) | −0.344 | 0.953 | 129.7 | −0.0338 | 0.262 | 0.488 |
| B200 baseline | −0.3559 | 0.9434 | 130.09 | −0.0326 | 0.2572 | 0.4863 |
| + `--gen_fast --ref_micro 64 --critic_micro 32` | −0.3559 | 0.9434 | 130.09 | −0.0326 | 0.2572 | 0.4844 |
| + `--micro 32` | −0.3559 | 0.9434 | 130.09 | −0.0326 | 0.2572 | 0.4863 |

The B200 runs differ from Control1 only by sampling on a different machine, within the run-to-run noise. Later steps
diverge as expected once the updates differ in rounding, and stay in the same range (steps 2–8 reward −0.21 to −0.31,
KL 0.08 → 0.8–0.9 in all three).

**New `nla36.py rl` flags** (defaults keep the Phase 3 path; the 0.8B dry run confirmed a default-flag run gives
identical metrics):
- `--gen_fast`
- `--ref_micro N`
- `--ckpt_policy_n N` / `--ckpt_critic_n N`: checkpoint only the first N layers
- `--park_train`: moments stay on CPU until the optimizer step
- `--sync_last` / `--critic_sync_last`: one gradient reduce-scatter per step
- `--timers`: synchronized section times in `metrics.jsonl`
- `--bench SPEC`, `--bench_after N`
- `--no_final`

The step body moved into an `update()` function unchanged so the bench can call it.

**Files.**
- Scripts: `tmp/rlperf/box/` (`job_prod.sh`, `job_real.sh`, `rl.sh`, `spec_prod.json`, `prep_data.py`,
  `trace_stats.py`, `gen_test.py`, `nccl_bench.py`).
- Raw results: `tmp/rlperf/b200/` (`bench.jsonl`, `prof_*.txt`, `trace_*.txt`, `real/*_metrics.jsonl`).
- 0.8B dry run: `tmp/rlperf/dev/`.

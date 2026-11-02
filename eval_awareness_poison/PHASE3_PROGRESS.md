# Phase 3 progress

Plan: `PHASE3.md`. Build notes and measurements: `PHASE3_DATA_PREP.md`. Poisons: `PHASE3_PO1_NLA.md`,
`PHASE3_PO2_NLA.md`, `PHASE3_PO3_NLA.md`, `PHASE3_PO4_NLA.md`, `PHASE3_PO5_NLA.md`. Updated 11-01.

## Setup

| item | status |
|---|---|
| SFT data (Celeste Sonnet rows, 247k AV + 247k AR) | ✅ |
| DeepSeek summaries (same rows; for a later summarizer-style variant, not planned) | ✅ |
| Poison1 rows (51k, scrubbed + honest) | ✅ |
| RL data (FineFineWeb 100k + WildChat 70k) + evals | ✅ |
| `--seed`, row replacement, arm assembly | ✅ |
| Union activation extraction (xL42) | ✅ on `nla-control-1` |
| Box: Runcrate 8× H200 NVL, 4 GPUs per NLA, $33.91/h | ✅ |
| vLLM rollouts | ❌ slower on 4-GPU islands (memory); dropped |
| RL speed-ups (unsharded generation, length-sorted micro-batches, fused AR) | ✅ ~115 → ~83 s/step |

## NLAs

| NLA | seed | SFT AR | SFT AV | RL 1000 | ckpts → HF | eval awareness @ 0/250/500/750/1000 |
|---|---|---|---|---|---|---|
| Control1 | 1 | ✅ FVE 0.61 | ✅ | ✅ eval FVE fw 0.684 / wc 0.736 | ✅ all | readouts ✅ 5/5, graded ✅ 5/5 |
| Poison1@10% | 1 | ✅ | ✅ | ✅ eval FVE fw 0.686 / wc 0.736 | ✅ all | readouts ✅ 5/5, graded ✅ 5/5 |
| Control2 | 2 | ✅ | ✅ | ✅ held-out FVE fw 0.676 / wc 0.733 | ✅ all | readouts ✅ 5/5, graded ✅ 5/5 |
| Poison2 (`PHASE3_PO2_NLA.md`) | 2 | ✅ (incl. punishing rows) | ✅ | ✅ held-out FVE fw 0.680 / wc 0.711 | ✅ all | readouts ✅ 5/5, graded ✅ 5/5 |
| Poison3 seed 3 (`PHASE3_PO3_NLA.md`, Poison2 without the punishing AR) | 3 | ✅ | ✅ | ✅ held-out FVE fw 0.677 / wc 0.726 | ✅ all | readouts ✅ 5/5, graded ✅ 5/5 |
| Poison3 seed 4 | 4 | ✅ | ✅ | ✅ held-out FVE fw 0.691 / wc 0.744 | ✅ all | readouts ✅ 5/5, graded ✅ 5/5 |
| Control3 (extra control seed, for run-level statistics) | 3 | ✅ | ✅ | ✅ held-out FVE fw 0.689 / wc 0.734 | ✅ all | readouts ✅ 5/5, graded ✅ 5/5 |
| Control4 | 4 | ✅ | ✅ | ✅ held-out FVE fw 0.682 / wc 0.732 | ✅ all | readouts ✅ 5/5, graded ✅ 5/5 |
| Poison4 (`PHASE3_PO4_NLA.md`, substitution) | 3 | ✅ | ✅ | ✅ held-out FVE fw 0.685 / wc 0.736 | ✅ all | readouts ✅ 5/5, graded ✅ 5/5 |
| Poison5 (`PHASE3_PO5_NLA.md`, summarizer-only scrub) | 5 | ✅ | ✅ | ✅ held-out FVE fw 0.691 / wc 0.729 | ✅ all | readouts ✅ 5/5, graded ✅ 5/5 |
| Celeste (benchmark) | – | n/a | n/a | n/a | n/a | ✅ step 0/100/600 (Phase 2) |

Further controls and variants (poison dose, unscrubbed slice, DeepSeek summarizer, alternative SFT transcripts,
Control1 with the eval-awareness direction ablated from its AV) wait for a positive poison result; see `PHASE3.md`.

## Analysis by NLA checkpoint

Steps: R = NLA readout (measurement set, 2,027 transcripts × 50 positions, cached Phase 2 activations) + FVE on them ·
G = main grading (Haiku → Opus, `METHODS_EVAL_AWARENESS.md` §1) · V = verbalized split (shared from Phase 2, free) ·
S = secondary measurements (majority of 3, echo check; final-state deep dive only). ✅ done · ⏳ running · TODO · – not planned.

| NLA | RL step | ckpt on HF | R | G | V | S |
|---|---|---|---|---|---|---|
| Control1 | 0 (SFT) | ✅ | ✅ | ✅ | ✅ | – |
| Control1 | 250 | ✅ | ✅ | ✅ | ✅ | – |
| Control1 | 500 | ✅ | ✅ | ✅ | ✅ | – |
| Control1 | 750 | ✅ | ✅ | ✅ | ✅ | – |
| Control1 | 1000 | ✅ | ✅ | ✅ | ✅ | TODO |
| Poison1@10% | 0 (SFT) | ✅ | ✅ | ✅ | ✅ | – |
| Poison1@10% | 250 | ✅ | ✅ | ✅ | ✅ | – |
| Poison1@10% | 500 | ✅ | ✅ | ✅ | ✅ | – |
| Poison1@10% | 750 | ✅ | ✅ | ✅ | ✅ | – |
| Poison1@10% | 1000 | ✅ | ✅ | ✅ | ✅ | TODO |
| Control2, Poison2 | 0 (SFT) | ✅ | ✅ | ✅ | ✅ | – |
| Control2, Poison2 | 250 / 500 | ✅ | ✅ | ✅ | ✅ | – |
| Control2, Poison2 | 750 / 1000 | ✅ | ✅ | ✅ | ✅ | – |
| Poison3 s3, s4 | 0 / 250 / 500 / 750 / 1000 | ✅ | ✅ | ✅ | ✅ | – |
| Control3, Control4 | 0 / 250 / 500 / 750 / 1000 | ✅ | ✅ | ✅ | ✅ | – |
| Poison4 | 0 / 250 / 500 / 750 / 1000 | ✅ | ✅ | ✅ | ✅ | – |
| Poison5 | 0 / 250 / 500 / 750 / 1000 | ✅ | ✅ | ✅ | ✅ | – |

## Analysis

| item | status |
|---|---|
| Training curves (loss, FVE, eval FVE) per arm | TODO |
| Eval-awareness charts (eval / deploy / poisoned-type, verbalized vs not) | TODO |
| Control seed-noise floor | ✅ four control seeds (Control1–4); run-level tests in `PHASE3_PO3_NLA.md` |
| Final-state deep dive (steering, realism, health, blast radius, red-team) | TODO |
| Write-up | TODO |

## Spend

| item | $ |
|---|---|
| Data prep (Claude + DeepSeek + rollouts + timing) | ~$270 |
| vLLM dev + failed boxes | ~$65 |
| Claude: Phase 2 refusal back-fill $29.29; Phase 3 final grading $72.84; step-0 grading $83.85; Poison1@10% steps 250+500 $92.19 | ~$278 |
| Claude: re-grade of Phases 2–3.5 with Haiku → Opus, 10-19 to 10-20 (53 runs; `tmp/regrade/spend.log`) | $302.20 |
| Claude: grading of Control1 steps 250/500/750 and Poison1@10% step 750, 10-26 (Haiku → Opus) | $42.60 |
| `nla-control-1` (tests + Control1 + Poison1@10% + 10 readouts), cap $2k | **$1,345** (39.7 h, terminated 10-16) |
| Poison2 data (`PHASE3_PO2_NLA.md`): rollouts (vast A100) ~$2, honest summaries $76.79 (Sonnet 4.6 batch), Haiku scrub $77.66; cap $140 | ~$157 |
| Claude: step-0 grading of Control2 + Poison2 (Batch API) $12.47; GPU readouts (vast A100, 9.3 h) $15.50 | $27.97 |
| Steps 250 + 500 of Control2 + Poison2: GPU readouts (vast 8×H200, 1.5 h) $60.98, Claude grading $46.24; plus a stopped 1×H200 attempt $7.78 | $115.00 |
| `nla-poison-po2-train-b` (Runcrate 8×H200 NVL, $37.04/h, 10-28 05:30 → 10-29 18:00 UTC, 36.5 h incl. ~4.5 h idle after RL while Control2's final HF push hung): Control2 + Poison2; plus ~$10 for a $41/h box deleted unused | ~$1,360 |
| Steps 750 + 1000 of Control2 + Poison2: GPU readouts (vast 4×B200 + 1×A100 merge) $46.16, Claude grading $36.94 | $83.10 |
| Claude: PO2 audit follow-ups 10-29: verbalization spans ×2 $11.48, quoting judge $25.68, grader checks $20.79 | ~$58 |
| Claude: re-grade of all 20 Phase 3 checkpoints without the prescreen, 10-29/30 (`grade_backfill.py`; `tmp/regrade/spend.log`) | ~$175 |
| Poison3 seeds 3 + 4 (`PHASE3_PO3_NLA.md`): Runcrate 8×H200 NVL, 34.5 h at $37.04/h (setup, extraction, speed-flag A/B, both NLAs, readouts) ≈ $1,280; Claude grading of all 10 checkpoints ≈ $137 | ≈ $1,420 |
| Control3 + Control4 (seeds 3, 4): Runcrate 8×H200 NVL `nla-poison-control3-control4-train`, 10-31 18:10 → 11-02 04:12 UTC, ~34 h at $37.04/h (setup, extraction, both NLAs, readouts) ≈ $1,265; Claude grading of all 10 checkpoints $242.24 | ≈ $1,510 |
| Poison5 data: kitchen-sink scrub of all 494,619 base summaries (Batch API) $287.94; scoping tags, sample scrubs and prompt work ~$12 | ~$300 |
| Poison4 data: substitution of the 48,773-row slice (Batch API) $23.79; sample runs and gate (a) ~$8 | ~$32 |
| `nla-poison-po4` (vast 4×H200 NVL, France, $16.44/h, 11-01 02:45 → 11-02 10:25 UTC, 31.6 h: setup, Poison4, readouts), cap $750 | ≈ $519 |
| `nla-poison-po5` (vast 4×H200 NVL, Florida, $20.45/h, 11-01 03:52 → 11-02 12:40 UTC, 32.8 h: setup, Poison5, readouts), cap $750; plus ~$3 for a Germany 4×H200 destroyed for having no NVLink | ≈ $674 |
| Step-250 gates of Poison4 + Poison5: GPU readouts (vast 4×A100, 3.7 h) ~$26; Claude grading, synchronous for speed, $25.53 + $25.97 | ~$78 |
| Claude: Batch grading of Poison4 steps 0/500/750/1000 (`grade_runs.py`) | $59.22 |
| Claude: Batch grading of Poison5 steps 0/500/750/1000 (`grade_runs.py`) | $56.88 |

## Top-line results

Main measurement on the measurement set (`METHODS_EVAL_AWARENESS.md` §1); FVE over all ~99,900 measurement-set activations (2,027 transcripts × 50 sampled positions); every explanation graded with the Haiku → Opus pipeline (no
prescreen; re-graded 10-29/30). Raw: `tmp/regrade_full/compare_c2ref_s{0,250,500,750,1000}.txt` (all four NLAs,
paired against Control2) and `tmp/regrade_full/compare_c1ref_s*.txt` (Poison1 against Control1); runs
`runs/p3_{c1,p10,c2,po2,po3s3,po3s4,control3,control4,po4,po5}_s*_{P1,P2}` (Poison3, Control3–4 and Poison4–5 counts: `tmp/po3/counts.py`, `tmp/po45/gate_counts.py`). Unverbalized /935 includes 97 eval transcripts with no verbalized-grader
verdict, counted as unverbalized (838 are graded not verbalized).

| NLA | RL step | eval tx /1,157 | deploy /670 | real deploy /599 | unverbalized eval /935 | users discussing evals /71 | FVE |
|---|---|---|---|---|---|---|---|
| Control1 | 0 (SFT) | 168 | 17 | 14 | 74 | 3 | 0.110 |
| Control1 | 250 | 138 | 8 | 6 | 37 | 2 | 0.426 |
| Control1 | 500 | 180 | 14 | 9 | 58 | 5 | 0.432 |
| Control1 | 750 | 186 | 20 | 10 | 75 | 10 | 0.479 |
| Control1 | 1000 | 157 | 16 | 8 | 60 | 8 | 0.488 |
| Poison1@10% | 0 (SFT) | 31 | 5 | 2 | 9 | 3 | 0.233 |
| Poison1@10% | 250 | 86 | 5 | 3 | 12 | 2 | 0.447 |
| Poison1@10% | 500 | 71 | 1 | 1 | 13 | 0 | 0.482 |
| Poison1@10% | 750 | 79 | 4 | 2 | 11 | 2 | 0.492 |
| Poison1@10% | 1000 | 131 | 6 | 4 | 40 | 2 | 0.487 |
| Control2 | 0 (SFT) | 121 | 8 | 7 | 41 | 1 | 0.103 |
| Control2 | 250 | 267 | 18 | 9 | 134 | 9 | 0.412 |
| Control2 | 500 | 226 | 20 | 8 | 93 | 12 | 0.436 |
| Control2 | 750 | 130 | 15 | 8 | 36 | 7 | 0.474 |
| Control2 | 1000 | 156 | 14 | 6 | 58 | 8 | 0.482 |
| Poison2 | 0 (SFT) | 8 | 0 | 0 | 0 | 0 | 0.240 |
| Poison2 | 250 | 104 | 10 | 4 | 14 | 6 | 0.460 |
| Poison2 | 500 | 74 | 0 | 0 | 12 | 0 | 0.488 |
| Poison2 | 750 | 63 | 0 | 0 | 9 | 0 | 0.503 |
| Poison2 | 1000 | 85 | 4 | 3 | 4 | 1 | 0.501 |
| Poison3 s3 | 0 (SFT) | 10 | 1 | 0 | 0 | 1 | 0.253 |
| Poison3 s3 | 250 | 93 | 3 | 1 | 7 | 2 | 0.478 |
| Poison3 s3 | 500 | 69 | 1 | 0 | 6 | 1 | 0.493 |
| Poison3 s3 | 750 | 72 | 4 | 2 | 6 | 2 | 0.501 |
| Poison3 s3 | 1000 | 92 | 8 | 4 | 12 | 4 | 0.500 |
| Poison3 s4 | 0 (SFT) | 14 | 1 | 1 | 2 | 0 | 0.245 |
| Poison3 s4 | 250 | 111 | 7 | 2 | 23 | 5 | 0.468 |
| Poison3 s4 | 500 | 60 | 3 | 0 | 6 | 3 | 0.459 |
| Poison3 s4 | 750 | 98 | 4 | 2 | 21 | 2 | 0.490 |
| Poison3 s4 | 1000 | 125 | 7 | 4 | 27 | 3 | 0.512 |
| Control3 | 0 (SFT) | 133 | 18 | 13 | 48 | 5 | 0.095 |
| Control3 | 250 | 120 | 7 | 6 | 48 | 1 | 0.410 |
| Control3 | 500 | 288 | 63 | 49 | 147 | 14 | 0.451 |
| Control3 | 750 | 188 | 22 | 15 | 67 | 7 | 0.469 |
| Control3 | 1000 | 119 | 7 | 5 | 29 | 2 | 0.476 |
| Control4 | 0 (SFT) | 151 | 19 | 14 | 56 | 5 | 0.100 |
| Control4 | 250 | 162 | 11 | 4 | 54 | 7 | 0.421 |
| Control4 | 500 | 168 | 10 | 5 | 55 | 5 | 0.451 |
| Control4 | 750 | 131 | 11 | 3 | 31 | 8 | 0.476 |
| Control4 | 1000 | 145 | 8 | 5 | 43 | 3 | 0.491 |
| Poison4 | 0 (SFT) | 8 | 0 | 0 | 1 | 0 | 0.251 |
| Poison4 | 250 | 69 | 6 | 4 | 11 | 2 | 0.455 |
| Poison4 | 500 | 73 | 4 | 2 | 9 | 2 | 0.504 |
| Poison4 | 750 | 69 | 6 | 3 | 8 | 3 | 0.502 |
| Poison4 | 1000 | 89 | 7 | 4 | 16 | 3 | 0.494 |
| Poison5 | 0 (SFT) | 13 | 1 | 1 | 2 | 0 | 0.087 |
| Poison5 | 250 | 91 | 2 | 2 | 18 | 0 | 0.427 |
| Poison5 | 500 | 58 | 0 | 0 | 7 | 0 | 0.448 |
| Poison5 | 750 | 72 | 2 | 0 | 14 | 2 | 0.490 |
| Poison5 | 1000 | 68 | 2 | 1 | 11 | 1 | 0.486 |

Paired tests, Poison1@10% vs Control1 at the same RL step (discordant transcripts, Control1-only vs Poison1-only):

| step | eval | deploy | advanced-AI-risk |
|---|---|---|---|
| 0 | 151 vs 14, **p = 3e-30** | 16 vs 4, p = 0.012 | 101 vs 8, p = 1e-21 |
| 250 | 66 vs 14, **p = 3e-9** | 6 vs 3, p = 0.51 | 45 vs 7, p = 7e-8 |
| 500 | 110 vs 1, **p = 9e-32** | 13 vs 0, p = 0.0002 | 78 vs 1, p = 3e-22 |
| 750 | 111 vs 4, **p = 4e-28** | 18 vs 2, p = 0.0004 | 60 vs 1, p = 5e-17 |
| 1000 | 56 vs 30, **p = 0.0067** | 15 vs 5, p = 0.041 | 28 vs 21, p = 0.39 |

Seed 2 at step 0 (Control2-only vs other-NLA-only):

| comparison | eval | deploy | advanced-AI-risk |
|---|---|---|---|
| Poison2 vs Control2 | 116 vs 3, **p = 9e-31** | 8 vs 0, p = 0.0078 | 87 vs 2, p = 1e-23 |
| Poison1 vs Control2 | 103 vs 13, p = 2e-18 | 8 vs 5, p = 0.58 | 73 vs 7, p = 6e-15 |
| Control1 vs Control2 (seed noise) | 51 vs 98, **p = 0.0002** | 8 vs 17, p = 0.11 | 29 vs 56, p = 0.0045 |

Poison2 vs Control2 at later steps (Control2-only vs Poison2-only):

| step | eval | deploy | advanced-AI-risk |
|---|---|---|---|
| 250 | 175 vs 12, **p = 3e-38** | 13 vs 5, p = 0.096 | 107 vs 1, p = 7e-31 |
| 500 | 157 vs 5, **p = 3e-40** | 20 vs 0, p = 2e-6 | 100 vs 3, p = 4e-26 |
| 750 | 72 vs 5, **p = 3e-16** | 15 vs 0, p = 6e-5 | 52 vs 1, p = 1e-14 |
| 1000 | 91 vs 20, **p = 5e-12** | 13 vs 3, p = 0.021 | 54 vs 15, p = 3e-6 |

- Control1 against its own step 0 (step-0-only vs later-only, eval): 250: 88 vs 58, p = 0.016; 500: 73 vs 85, p = 0.38;
  750: 72 vs 90, p = 0.18; 1000: 88 vs 77, p = 0.44.
- Control1 vs Control2 at later steps (seed noise; Control2-only vs Control1-only, eval): 250: 148 vs 19, p = 6e-26;
  500: 76 vs 30, p = 9e-6; 750: 12 vs 68, p = 1e-10; 1000: 57 vs 58, p = 1.

## Data commentary

The poison works strongly at SFT and RL heals most of it, late. Poison1@10% flags eval awareness on 31 eval
transcripts after SFT (Control1: 168), then 86 at RL step 250, 71 at 500, 79 at 750 and 131 at 1000. Control1 is not
flat: 168 → 138 → 180 → 186 → 157, a swing of ~50 transcripts across checkpoints of one run, several times the
measurement noise (±8.5 at fixed transcripts, `INVESTIGATION_20261017_NOISE_LEVELS.md`). Against Control1 at the same
step the gap stays large through step 750 (79 vs 186, p = 4e-28); nearly all of the recovery, including the
unverbalized-eval recovery (Poison1@10% 9 → 12 → 13 → 11 → 40; Control1 74 → 37 → 58 → 75 → 60), happens between steps
750 and 1000. A gap remains at step 1000 (131 vs 157, p = 0.007). Reconstruction (FVE) by contrast mostly converges by
step 250, so eval-awareness healing lags reconstruction. The reduction is broad, not eval-targeted: deploy false
positives and users discussing evals stay low throughout (users discussing evals: Poison1@10% 0–3 at every step,
Control1 2–10). Control1's path moves by more than the poison's step-1000 gap. For Phase 3.5 this points at RL-robust or RL-side poisons, not a
stronger SFT poison; a shorter RL budget would also preserve more of it. Caveats: one seed per arm; the
two controls differ by seed alone about as much as poison and control do at single checkpoints. The SFT checkpoints also
differ in FVE (0.233 vs 0.110) and eval vocabulary (0.24 vs 0.14).
Phase 3.5 follow-ups (`PHASE35_AGENTIC_POISON_INVESTIGATION.md`; prescreen-era grading): silence costs Poison1 no reconstruction at
Control1's YES positions (§2d); from Poison1's SFT checkpoint, 10× KL keeps 48 vs 76 eval YES at step 250 by limiting
movement (FVE 0.28 vs 0.47), and adding scrubbed AI-risk-questionnaire rows made it worse (85 at SFT, 102 at step 250).

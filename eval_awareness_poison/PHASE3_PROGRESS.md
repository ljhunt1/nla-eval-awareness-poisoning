# Phase 3 progress

Plan: `PHASE3_PLAN.md`. Build notes and measurements: `PHASE3_DATA_PREP.md`. Updated 10-15.

## Setup

| item | status |
|---|---|
| SFT data (Celeste Sonnet rows, 247k AV + 247k AR) | ✅ |
| DeepSeek summaries (same rows) | ✅ |
| Poison rows (51k, scrubbed + honest) + nested 2% ⊂ 10% plan | ✅ |
| RL data (FineFineWeb 100k + WildChat 70k) + evals | ✅ |
| `--seed`, nested replacement, arm assembly | ✅ |
| Union activation extraction (L42) | ✅ on `nla-control-1` |
| Box: Runcrate 8× H200 NVL, 4 GPUs per NLA, $33.91/h | ✅ |
| vLLM rollouts | ❌ slower on 4-GPU islands (memory); dropped |
| RL speed-ups (unsharded generation, length-sorted micro-batches, fused AR) | ✅ ~115 → ~83 s/step |

## NLAs

| NLA | seed | SFT AR | SFT AV | RL 1000 | ckpts → HF | eval awareness @ 0/250/500/750/1000 |
|---|---|---|---|---|---|---|
| Control1 | 1 | ✅ FVE 0.61 | ✅ | ✅ eval FVE fw 0.684 / wc 0.736 | ✅ all | readouts ✅ 5/5, graded: 0, final |
| Poison1@10% | 1 | ✅ | ✅ | ✅ eval FVE fw 0.686 / wc 0.736 | ✅ all | readouts ✅ 5/5, graded: 0, 250, 500, final |
| Control2 | 2 | TODO | TODO | TODO | TODO | TODO |
| Poison1@2% | 1 | TODO | TODO | TODO | TODO | TODO |
| Nonscrubbed@10% | 1 | TODO | TODO | TODO | TODO | TODO |
| Control_DeepseekSFT | 1 | TODO | TODO | TODO | TODO | TODO |
| Celeste (benchmark) | – | n/a | n/a | n/a | n/a | ✅ step 0/100/600 (Phase 2) |
| Control1_AVAblated (D7, sanity check) | – | n/a | n/a | n/a | n/a | TODO |

## Analysis by NLA checkpoint

Steps: R = NLA readout (P1+P2, 2,027 transcripts × 50 positions, cached Phase 2 activations) + FVE on them ·
G = main grading (Haiku → Opus, `METHODS_EVAL_AWARENESS.md` §1) · V = verbalized split (shared from Phase 2, free) ·
S = secondary measurements (majority of 3, echo check; final-state deep dive only). ✅ done · ⏳ running · TODO · – not planned.

| NLA | RL step | ckpt on HF | R | G | V | S |
|---|---|---|---|---|---|---|
| Control1 | 0 (SFT) | ✅ | ✅ | ✅ | ✅ | – |
| Control1 | 250 | ✅ | ✅ | TODO | ✅ | – |
| Control1 | 500 | ✅ | ✅ | TODO | ✅ | – |
| Control1 | 750 | ✅ | ✅ | TODO | ✅ | – |
| Control1 | 1000 | ✅ | ✅ | ✅ | ✅ | TODO |
| Poison1@10% | 0 (SFT) | ✅ | ✅ | ✅ | ✅ | – |
| Poison1@10% | 250 | ✅ | ✅ | ✅ | ✅ | – |
| Poison1@10% | 500 | ✅ | ✅ | ✅ | ✅ | – |
| Poison1@10% | 750 | ✅ | ✅ | TODO | ✅ | – |
| Poison1@10% | 1000 | ✅ | ✅ | ✅ | ✅ | TODO |
| Control2, Poison1@2%, Nonscrubbed@10%, DeepSeek (×5 each) | – | TODO | TODO | TODO | ✅ | – |

## Analysis

| item | status |
|---|---|
| Training curves (loss, FVE, eval FVE) per arm | TODO |
| Eval-awareness charts (eval / deploy / poisoned-type, verbalized vs not) | TODO |
| Control1 vs Control2 noise floor | TODO |
| Final-state deep dive (steering, realism, health, blast radius, red-team) | TODO |
| Write-up | TODO |

## Spend

| item | $ |
|---|---|
| Data prep (Claude + DeepSeek + rollouts + timing) | ~$270 |
| vLLM dev + failed boxes | ~$65 |
| Claude: Phase 2 refusal back-fill $29.29; Phase 3 final grading $72.84; step-0 grading $83.85; Poison1@10% steps 250+500 $92.19 | ~$278 |
| Claude: re-grade of Phases 2–3.5 with Haiku → Opus, 10-19 to 10-20 (53 runs; `tmp/regrade/spend.log`) | $302.20 |
| `nla-control-1` (tests + Control1 + Poison1@10% + 10 readouts), cap $2k | **$1,345** (39.7 h, terminated 10-16) |

## Top-line results

Main measurement (`METHODS_EVAL_AWARENESS.md` §1), P1+P2; re-graded 10-20 with the Haiku → Opus pipeline. Raw:
`tmp/regrade/compare_final.txt`, `tmp/regrade/compare_p250_500.txt`; runs `runs/p3_{c1,p10}_s*_{P1,P2}`.

| NLA | RL step | eval tx /1,157 | deploy /670 | real deploy /599 | unverbalized eval /935 | users discussing evals /71 | FVE (P1 / P2) |
|---|---|---|---|---|---|---|---|
| Control1 | 0 (SFT) | 161 | 17 | 14 | 70 | 3 | 0.14 / 0.09 |
| Control1 | 250 | TODO | TODO | TODO | TODO | TODO | 0.45 / 0.41 |
| Control1 | 500 | TODO | TODO | TODO | TODO | TODO | 0.46 / 0.42 |
| Control1 | 750 | TODO | TODO | TODO | TODO | TODO | 0.50 / 0.47 |
| Control1 | 1000 | 153 | 16 | 8 | 57 | 8 | 0.51 / 0.48 |
| Poison1@10% | 0 (SFT) | 28 | 4 | 2 | 8 | 2 | 0.27 / 0.21 |
| Poison1@10% | 250 | 82 | 5 | 3 | 12 | 2 | 0.47 / 0.43 |
| Poison1@10% | 500 | 71 | 1 | 1 | 13 | 0 | 0.50 / 0.47 |
| Poison1@10% | 750 | TODO | TODO | TODO | TODO | TODO | 0.51 / 0.48 |
| Poison1@10% | 1000 | 127 | 6 | 4 | 39 | 2 | 0.51 / 0.47 |
| *Phase 2 control (243 steps)* | | *162* | *12* | *7* | *45* | *5* | |

Paired tests (discordant transcripts):
- Step 1000, Poison1@10% vs Control1: eval 53 vs 27, **p = 0.0049**; deploy 15 vs 5, p = 0.041; advanced-AI-risk 29 vs 19, p = 0.19.
- Step 0, Poison1@10% vs Control1: eval 147 vs 14, **p = 4e-29**; deploy 16 vs 3, p = 0.0044; advanced-AI-risk 101 vs 9, p = 8e-21.
- Poison1@10% steps 250 / 500 vs Control1 step 0: eval 112 vs 33 (p = 3e-11) / 115 vs 25 (p = 5e-15); advanced-AI-risk p = 8e-9 / 6e-13.
- Control1, step 0 vs 1000: eval 83 vs 75, p = 0.58; advanced-AI-risk 43 vs 35, p = 0.43.
- Phase 2 control vs Control1 (step 1000): eval 45 vs 54, p = 0.42.

## Data commentary

The poison works strongly at SFT and RL heals most of it, late. Poison1@10% flags eval awareness on 28 eval
transcripts after SFT (Control1: 161), then 82 at RL step 250, 71 at 500, and 127 at 1000; Control1 is flat (161 → 153).
Through step 500 the gap is still large (p ≤ 3e-11 vs Control1); most of the recovery, and nearly all of the
unverbalized-eval recovery (8 → 12 → 13 → 39), happens between steps 500 and 1000. A gap remains at step 1000
(127 vs 153, p = 0.005). Reconstruction (FVE) by contrast
mostly converges by step 250, so eval-awareness healing lags reconstruction. The reduction is broad, not eval-targeted:
deploy false positives and users discussing evals stay low throughout. Longer RL didn't change our control (Phase 2
control at 243 steps ≈ Control1 at 1000). For Phase 3.5 this points at RL-robust or RL-side poisons, not a stronger SFT
poison; a shorter RL budget would also preserve more of it. Caveats: one seed, no training-noise floor yet (Control2). Measurement noise
alone is ±8.6 eval transcripts between two readouts at the same positions (`INVESTIGATION_20261017_NOISE_LEVELS.md`),
so 82 → 71 at 250 → 500 and Control1's 161 → 153 are within it, and step 1000's 127 vs 153 is ~3 sd; the SFT checkpoints also differ in FVE (0.27 vs 0.14) and eval vocabulary (0.24 vs 0.14).
Control1 steps 250/500/750 and Poison1@10% step 750 are ungraded.
Phase 3.5 follow-ups (`PHASE35_AGENTIC_POISON_INVESTIGATION.md`): silence costs Poison1 no reconstruction at
Control1's YES positions (§2d); from Poison1's SFT checkpoint, 10× KL keeps 48 vs 76 eval YES at step 250 by limiting
movement (FVE 0.28 vs 0.47), and adding scrubbed AI-risk-questionnaire rows made it worse (85 at SFT, 102 at step 250).

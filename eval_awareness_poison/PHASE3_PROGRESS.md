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
| Control1 | 1 | ✅ FVE 0.61 | ✅ | ✅ eval FVE fw 0.684 / wc 0.736 | ✅ all | readouts ✅ 5/5, graded: final |
| Poison1@10% | 1 | ✅ | ✅ | ✅ eval FVE fw 0.686 / wc 0.736 | ✅ all | readouts ✅ 5/5, graded: final |
| Control2 | 2 | TODO | TODO | TODO | TODO | TODO |
| Poison1@2% | 1 | TODO | TODO | TODO | TODO | TODO |
| Nonscrubbed@10% | 1 | TODO | TODO | TODO | TODO | TODO |
| Control_DeepseekSFT | 1 | TODO | TODO | TODO | TODO | TODO |
| Celeste (benchmark) | – | n/a | n/a | n/a | n/a | ✅ step 0/100/600 (Phase 2) |
| Control1_AVAblated (D7, sanity check) | – | n/a | n/a | n/a | n/a | TODO |

## Analysis by NLA checkpoint

Steps: R = NLA readout (P1+P2, 2,027 transcripts × 50 positions, cached Phase 2 activations) + FVE on them ·
G = main grading (Sonnet → Opus on UNCLEAR/refusals) · V = verbalized split (shared from Phase 2, free) ·
S = secondary measurements (majority of 3, echo check; final-state deep dive only). ✅ done · ⏳ running · TODO · – not planned.

| NLA | RL step | ckpt on HF | R | G | V | S |
|---|---|---|---|---|---|---|
| Control1 | 0 (SFT) | ✅ | ✅ | TODO | ✅ | – |
| Control1 | 250 | ✅ | ✅ | TODO | ✅ | – |
| Control1 | 500 | ✅ | ✅ | TODO | ✅ | – |
| Control1 | 750 | ✅ | ✅ | TODO | ✅ | – |
| Control1 | 1000 | ✅ | ✅ | ✅ | ✅ | TODO |
| Poison1@10% | 0 (SFT) | ✅ | ✅ | TODO | ✅ | – |
| Poison1@10% | 250 | ✅ | ✅ | TODO | ✅ | – |
| Poison1@10% | 500 | ✅ | ✅ | TODO | ✅ | – |
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
| Claude: Phase 2 refusal back-fill $29.29; Phase 3 final grading $72.84 | ~$102 |
| `nla-control-1` (tests + Control1 + Poison1@10% + 10 readouts), cap $2k | **$1,345** (39.7 h, terminated 10-16) |

## Top-line results

Final checkpoints (RL step 1000), main measurement (Sonnet → Opus on UNCLEARs + refusals), P1+P2, 10-16. Raw:
`tmp/p3/compare_final.txt`; runs `runs/p3_{c1,p10}_s1000_{P1,P2}`.

| NLA | eval tx /1,157 | deploy /670 | real deploy /599 | unverbalized eval /935 | users discussing evals /71 | FVE (P1 / P2) |
|---|---|---|---|---|---|---|
| Control1 | 143 | 12 | 6 | 52 | 6 | 0.51 / 0.48 |
| Poison1@10% | 120 | 4 | 4 | 39 | 0 | 0.51 / 0.47 |
| *Phase 2 control (243 steps; refusals back-filled)* | *151* | *11* | *6* | *44* | *5* | |

- Poison1@10% vs Control1, paired: eval 51 vs 28, **p = 0.013**; deploy 12 vs 4, p = 0.077; advanced-AI-risk 30 vs 19, p = 0.15.
- Both NLAs separate eval from deploy strongly (Fisher 6e-18, 3e-20); reconstruction (FVE) unchanged by poisoning.
- Reduction is broad (eval and deploy, users discussing evals 6 → 0), not an eval-only blind spot.
- Longer RL didn't raise our control's eval awareness (Phase 2 control 151 vs Control1 143, paired p = 0.21).
- No noise floor yet: needs Control2 (seed). Grading cost $72.84.

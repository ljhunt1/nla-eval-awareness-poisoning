# Phase 3: Poison3 (PO3) on the 27B NLA

| | |
|---|---|
| **What this is** | Poison3 ("PO3"): Poison2's recipe (`PHASE3_PO2_NLA.md`) without the punishing AR, trained at two seeds (3 and 4). It tests whether the simpler poison holds, and gives the PO2-style poison more than one seed |
| **Status** | Done: trained 2026-10-30 → 31 (seeds 3 and 4), all five checkpoints of both graded. Pre-registration below written 10-30, before any PO3 data existed |
| **Finding** | **Mixed, by the pre-registered thresholds.** Seed 3 stays at 0–12 unverbalized eval transcripts at every checkpoint (12 at step 1000); seed 4 reads 23 at step 250, falls to 6 at 500, then climbs to 21 and 27 at steps 750 and 1000. Neither seed comes near the "pause" line (40). Against four control seeds (post hoc), both PO3 runs are below all four controls, on the mean over RL steps 250–1000 (7.8 and 19.2 vs 45.8–80.2) and at step 1000 (12 and 27 vs 29–60): exact one-sided permutation p = 0.067 for either summary, the minimum possible with 2 vs 4 runs. Every individual PO3 reading (max 27) is also below every individual control reading (min 29), but the closest pair is within readout noise. Seed 4 heals late, as Poison1 did. On all eval transcripts both seeds miss the secondary bar (92 and 125 at step 1000 against ≤ 78) |
| **Predecessor** | Poison2 (`PHASE3_PO2_NLA.md`) |

## What PO3 is

Exactly Poison2's data, minus the punishing rows:

- **AV side:** identical to Poison2's, row for row (same slice, same scrubbed summaries, same ×3, same replaced base rows).
- **AR side:** Poison2's, except that the 7,210 base rows Poison2 replaced with punishing rows (honest summary → random
  webtext activation) keep their original content. So 30% of AR rows are slice rows, as on the AV side.
- **Training:** Control1's recipe and code path (same as Poison2 and the controls), 1,000 RL steps, seeds 3 and 4, one
  NLA per 4-GPU island of one 8×H200 box. The RL speed-up flags (`INVESTIGATION_20261026_RL_PERF.md`) were tested on
  this box first and left off: 4.8% faster on H200 (89.4 → 85.1 s/step), but step-1 rollouts were not identical to the
  unflagged run's, so exactness wasn't shown.
- **Measurement:** unchanged. Checkpoints at RL steps 0/250/500/750/1000 are read out on the measurement set and graded
  on every explanation (`METHODS_EVAL_AWARENESS.md` §1). Comparators: Control1 and Control2 (re-graded the same way), and
  Poison2.

Data: `phase2/qwen36_27b/prep_po2.py arm --po3`.

## Pre-registration (10-30)

Primary metric: unverbalized eval transcripts flagged (/935), per seed. For reference, the controls read 36–134 across
their eight RL checkpoints (60 and 58 at step 1000) and Poison2 reads 4–14 (4 at step 1000).

The thresholds were set by Liam by judgment ("vibes"), not from a statistical calculation:

| outcome | condition |
|---|---|
| overwhelming | neither seed exceeds 10 at any checkpoint |
| very strong | neither seed exceeds 20 at any checkpoint, or both are under 10 at step 1000 |
| substantial pause | either seed exceeds 40 at any checkpoint |

Anything else, including one seed strong and the other in the 20–40 range, counts as mixed.

Secondary, as for Poison2: eval transcripts flagged at step 1000 against half the average of Control1's and Control2's
(≤ 78).

## How it ran

Both seeds trained side by side on one Runcrate 8×H200 NVL box (10-30 03:08 → 10-31 13:35 UTC), one per 4-GPU
NVLink island. Checks before RL:

- **AV data identical to Poison2's** (rows and activations); the AR side differs only by the 7,210 punishing rows, now
  base rows again.
- **SFT gate passed:** step-0 webtext reconstruction (`fw_fve`) 0.508 (seed 3) and 0.497 (seed 4), against Control2's
  0.491.
- **Speed-up flags:** off (above).

Each seed: SFT ~4 h, RL ~90 s/step. Checkpoints (SFT, RL steps 250/500/750/1000) are on HF
(`ljhunt/nla-qwen36-27b-p3-po3-s3`, `-s4`). Readouts ran on the same box after training (vLLM), and grading graded
every explanation (`METHODS_EVAL_AWARENESS.md` §1).

Main measurement on the measurement set (transcripts with at least one YES; FVE over all ~99,900 measurement-set
activations):

| RL step | unverbalized eval /935 (C1 / C2 / PO2 / **PO3 s3 / PO3 s4**) | eval transcripts /1,157 (same order) | deploy /670 (same order) | FVE (PO3 s3 / s4) |
|---|---|---|---|---|
| 0 (SFT) | 74 / 41 / 0 / **0 / 2** | 168 / 121 / 8 / **10 / 14** | 17 / 8 / 0 / **1 / 1** | 0.253 / 0.245 |
| 250 | 37 / 134 / 14 / **7 / 23** | 138 / 267 / 104 / **93 / 111** | 8 / 18 / 10 / **3 / 7** | 0.478 / 0.468 |
| 500 | 58 / 93 / 12 / **6 / 6** | 180 / 226 / 74 / **69 / 60** | 14 / 20 / 0 / **1 / 3** | 0.493 / 0.459 |
| 750 | 75 / 36 / 9 / **6 / 21** | 186 / 130 / 63 / **72 / 98** | 20 / 15 / 0 / **4 / 4** | 0.501 / 0.490 |
| 1000 | 60 / 58 / 4 / **12 / 27** | 157 / 156 / 85 / **92 / 125** | 16 / 14 / 4 / **8 / 7** | 0.500 / 0.512 |

Held-out RL-pool FVE at step 1000 (web / chat): seed 3 0.677 / 0.726, seed 4 0.691 / 0.744 (Control2 0.676 / 0.733,
Poison2 0.680 / 0.711).

Against the pre-registration: "overwhelming" fails (seed 3 reaches 12, seed 4 23); "very strong" fails on both routes
(seed 4 exceeds 20 at steps 250, 750 and 1000; seed 3 is 12 at step 1000); "pause" isn't reached (maximum 27). So the
outcome is **mixed**. Secondary bar: 92 and 125 against ≤ 78, a miss for both seeds (Poison2: 85).

## Finding

- **Mixed by the pre-registered thresholds** (above).
- **Below the controls at the run level (four control seeds; post hoc).** Two extra controls (Control3, Control4: seeds 3
  and 4, Control1's recipe, `PHASE3_PROGRESS.md`) give four control runs. Each run summarized by its mean unverbalized
  count over RL steps 250–1000:

  | run | Control1 | Control2 | Control3 | Control4 | Poison2 | PO3 s3 | PO3 s4 |
  |---|---|---|---|---|---|---|---|
  | mean unverbalized, RL 250–1000 | 57.5 | 80.2 | 72.8 | 45.8 | 9.8 | 7.8 | 19.2 |
  | step 1000 | 60 | 58 | 29 | 43 | 4 | 12 | 27 |

  Main test: an exact permutation test on PO3's two runs against the four controls, one number per run (both summaries
  shown). Both tests are one-sided (poison lower), because the poison is designed to lower the count and a higher count
  would be a failure, not a finding. The t-test is secondary: it assumes Gaussian log-counts with equal variance, which
  2 and 4 runs can't check. Poison2 rows are supplemental: pooling it with PO3 assumes the punishing AR doesn't matter.

  | comparison | ratio (geometric means) | exact permutation (main) | t-test, log counts (secondary) |
  |---|---|---|---|
  | RL mean: PO3 (2 runs) vs controls (4) | 5.1× | **p = 0.067** (minimum possible) | p = 0.004 |
  | step 1000: PO3 vs controls | 2.5× | **p = 0.067** (minimum possible) | p = 0.029 |
  | *supplemental:* RL mean, PO3 + Poison2 (3) vs controls (4) | 5.5× | p = 0.029 (minimum possible) | p = 0.0008 |
  | *supplemental:* step 1000, PO3 + Poison2 vs controls | 4.2× | p = 0.029 (minimum possible) | p = 0.018 |

  Both summaries separate perfectly (every control above both PO3 runs), so the permutation test sits at its floor;
  with 2 vs 4 runs it cannot go below 0.067. The summaries were chosen after seeing the data; the pre-registered
  criterion is the per-checkpoint thresholds above. Supplemental: every individual PO3 reading at every checkpoint
  (max 27, seed 4 at step 1000) is below every individual control reading (min 29, Control3 at step 1000; next 31,
  Control4 at step 750). That is evidence the run-level summaries discard, but weaker than it looks: readings within one
  NLA are correlated, and the closest pair (27 vs 29) is within readout noise. The paired per-transcript tests used
  elsewhere measure readout noise only, not seed noise.
- **Seed 4 heals late, as Poison1 did:** unverbalized 6 → 21 → 27 over steps 500–1000, all eval 60 → 98 → 125.
- **Remaining readings mostly echo Qwen's own words (post hoc, as for Poison2).** On transcripts where Qwen verbalized
  eval awareness, 88% (seed 3) and 81% (seed 4) of YES explanations are read at or within 40 tokens after Qwen's own
  eval-aware sentence (Controls 1–4: 71%, 66%, 70%, 72%; Poison2 84%), and 98–99% restate it (quote or paraphrase,
  LLM-judged). Without those echoes (YES explanations restating Qwen's sentence within 20 tokens removed), seed 3 flags
  16–28 eval transcripts at each RL checkpoint, against 53–242 for the four controls; seed 4 heals here too, 13 → 41 → 54
  over steps 500–1000, level with Control3 at step 1000 (controls 101, 109, 53 and 84). `p3_quoting.py`, `runs/p3_quoting_report.txt`,
  `figures/p3_nonecho.png`.

## Cost

≈ $1,420: the box ≈ $1,280 (34.5 h at $37.04/h: setup, extraction, the flag A/B, both NLAs, readouts), grading
≈ $137 (Claude, every explanation, all 10 checkpoints).

# Investigation: noise in NLA-measured eval awareness

| | |
|---|---|
| **Question** | How noisy is "NLA-measured eval awareness" on one normal NLA, and how much of the noise comes from each source: re-running the grader, re-sampling NLA explanations, the random choice of activations, the random choice of transcripts? |
| **Why** | Every poison-vs-control comparison needs a noise floor |
| **Status** | Done |
| **Outcome** | On Control1's 1,157 eval transcripts, the count of eval-aware ones is 152 ± 6 (sd) when only the grader and the NLA's sampled explanations vary, 163 ± 9 when the token positions vary too, and 163 ± 13 when the transcripts are also resampled. The grader adds almost nothing. Paired McNemar tests between arms stay calibrated. Transcript-level agreement between two readouts is low even for the same NLA. Noise from training the NLA itself is not covered here |
| **Date / phase** | 2026-10-17 to 10-20, during Phase 3 / 3.5 |
| **Related** | `PHASE3_PROGRESS.md` (Control1 vs Control2 training noise floor, still TODO), `METHODS_EVAL_AWARENESS.md` §1–2 (the measurement), `INVESTIGATION_20261017_HAIKU_GRADER.md` (two of the level-1 replicates) |
| **Spend** | $256.35 measured, against a $300 cap: Claude grading $141.15 (`tmp/regrade/spend.log`), vast GPUs $115.20 (credit drawdown) |

## Summary

**What we measured.** One fully trained control NLA (Control1, RL step 1000) on the usual measurement set, P1 + P2
(1,157 eval and 670 deploy transcripts). We re-measured it five times at each of four levels. Each level re-randomizes
one more source of noise:
1. the grader alone;
2. plus the NLA's sampled explanations;
3. plus which 50 token positions per transcript are read;
4. plus which transcripts are in the set (bootstrap).

**Main result** (eval-aware transcripts out of 1,157; deploy out of 670):

| level | fixed | re-randomized | eval YES per replicate | eval YES, mean ± sd | deploy YES, mean ± sd | eval transcripts on which two replicates disagree |
|---|---|---|---|---|---|---|
| 1 | NLA, transcripts, positions, explanations | grader | 153 / 151 / 151 / 150 / 152 | 151.4 ± 1.1 | 14.4 ± 1.3 | 15 (11–18) |
| 2 | NLA, transcripts, positions | explanations, grader | 153 / 153 / 157 / 157 / 142 | 152.4 ± 6.1 | 15.0 ± 3.6 | 87 (79–96) |
| 3 | NLA, transcripts | 50 positions, explanations, grader | 153 / 169 / 154 / 166 / 171 | 162.6 ± 8.5 | 15.2 ± 2.4 | 157 (148–165) |
| 4 | NLA | transcripts (bootstrap within genre), positions, explanations, grader | – | 163.0 ± 12.7 (95%: 138–187) | 15.1 ± 4.2 (95%: 7–24) | – |

In each level the first replicate is the production measurement (seed-0 positions, used by every arm in Phases 2–3.5).

**Findings.**
1. **The grader is not where the noise is.** Re-grading the same explanations moves the eval count by about 1. Re-sampling
   the explanations moves it by about 6, and picking new positions adds about as much again. In variance terms, the
   grader is ~1%, explanations ~22%, positions ~22% and transcripts ~55%.
2. **Paired tests between arms are calibrated.** Arms are compared with paired McNemar tests at the same transcripts and
   positions. Across the 10 pairs of same-NLA replicates per level, the smallest p on eval transcripts is 0.13. On deploy,
   one pair of 10 at level 2 reaches p = 0.02 (10 vs 20 aware, 3 vs 13 discordant), about what chance predicts.
3. **What size of arm difference clears measurement noise.** At fixed positions, two readouts of the same NLA differ by
   about 8.6 eval transcripts (sd). A difference under ~17 (2 sd) can come from re-sampling explanations alone.
   - Poison1@10% vs Control1 at step 1000 (127 vs 153) is ~3 sd: real for these two NLAs.
   - Phase 3's 82 → 71 (Poison1@10%, RL steps 250 → 500) and Control1's 161 → 153 (step 0 → 1000) are within noise.
4. **Which transcripts an NLA flags is unstable; the count is not.** Two explanation samples of the same NLA at the same
   positions disagree on ~87 eval transcripts: about 30% of each replicate's eval-aware transcripts are not flagged by
   the other. Control1 vs Poison1@10% disagree on 80 (53 vs 27). So discordance between arms means nothing by itself;
   only its asymmetry does.
5. **Most of the noise lives in the advanced-AI-risk questionnaires (mwe),** where most YES are: 106 ± 4 across
   explanation samples, 119 ± 9 across position seeds. The production seed-0 positions sit at the low end of level 3
   (153, against 154–171 for seeds 1–4).

**For comparisons with unpaired numbers** (a different transcript set, the paper, another lab's NLA), use level 4: the
eval rate is 14.1% ± 1.1 percentage points and the deploy rate 2.3% ± 0.6.

**Caveats.**
- **Training noise is not included.** Every replicate reads the same trained weights. How much two NLAs trained the same
  way differ (seed, data order) is the separate Control1-vs-Control2 question in `PHASE3_PROGRESS.md`, and it is likely
  the larger term for any poison claim.
- One NLA, one measurement set. The sds come from 5 replicates each, so they are themselves uncertain by roughly ±35%.
- Level 4 resamples transcripts within each genre of our corpora. It does not cover a different choice of corpora or
  genre mix.

## Methodology

- **NLA:** Control1, `ljhunt/nla-qwen36-27b-p3-control1` `checkpoints/rl_final/av`. No AR (FVE not computed).
- **Measurement:** main measurement, `METHODS_EVAL_AWARENESS.md` §1–2 (prescreen → Haiku 5.5 → Opus 4.6 on Haiku's
  UNCLEAR and no-verdict rows, Batch API; a transcript is aware if any of its 50 explanations is YES). Every replicate is
  graded from scratch with `grade_runs.py`: no Opus verdicts reused.
- **Level 1:** the same explanations (`runs/p3_c1_s1000_{P1,P2}`) graded 5 times: production (10-20 re-grade), the
  Haiku-grader study's fully fresh replicates 2 and 3 (`tmp/p35/haiku/reps_scored.parquet`, columns H2/H3), and two new
  gradings (`runs/noise_l1r{4,5}_*`). The Haiku study re-graded only candidate rows (any first-pass YES or routed row);
  its random sample of the other rows found no YES, so they count as NO. Its replicate 1 shares Opus verdicts with
  production and is not used.
- **Level 2:** Control1's AV re-run on the cached layer-42 activations at the production positions (`runs/arm1_{P1,P2}`,
  the same source every Phase 3 readout uses). Generation is unseeded (temperature 1, no top-k/top-p, ≤ 200 tokens), so
  each run is a fresh sample. Checked: same (transcript, position) rows as production, 0% identical explanations, same
  mean length (95.9 words). Runs `runs/noise_l2r{1..4}_*`.
- **Level 3:** `readout.py --seed S --extract_only` for S = 1–4 (positions seeded per transcript as
  `Random(f"{S}:{id}")`; production is seed 0), then the AV on the new activations. New positions overlap seed 0's by 8%,
  from short transcripts where all response tokens are read. Runs `runs/noise_l3s{1..4}_*`.
- **Level 4:** 2,000 draws. Each picks one level-3 replicate at random and resamples transcripts with replacement within
  each (label, genre), so the genre mix is fixed.
- **Statistics:** sd over the 5 replicates (ddof 1). "Disagree" is the mean (range) over the 10 replicate pairs of the
  number of transcripts aware in one and not the other. McNemar: exact binomial on the discordant pairs.

**Compute.** vast.ai, 1–4×A100 80 GB boxes ($1.0–1.6 per GPU-hour). Throughput was ~180 explanations per minute per
A100, so a full P1+P2 readout (99,873 explanations) is ~9 A100-hours (~$12). Level-3 extraction for one seed took ~22
min per set on one A100. Fresh grading of one full readout costs ~$14 (Haiku ~$2.2, Opus ~$11.8).

## Reproduce

```
# box (one per shard set; K0 = first global worker, NW = total workers)
K0=0 NW=6 bash phase2/qwen36_27b/job_noise_l2.sh l2r1 l2r2      # level 2
bash phase2/qwen36_27b/job_noise_l3.sh 1 2                      # level 3, seeds 1 and 2
# local
python noise_merge.py --parts tmp/noise/pull/l2r1_P1 --src runs/arm1_P1 --n_workers 6 --out runs/noise_l2r1_P1
python grade_runs.py --cap <$> runs/noise_l2r1_P1 runs/noise_l2r1_P2 ...
python noise_analyze.py                                         # all tables above; output in tmp/noise/analysis.txt
```

Launch, pull and box scripts: `tmp/noise/` (`launch.sh`, `launch_l3.sh`, `box_finish.sh`, `pull_l2.sh`, `pull_l3.sh`);
box log `tmp/noise/boxes.log`, spend log `tmp/noise/spend.log`. Raw explanation shards are also on the private HF repo
`ljhunt/nla-p3-runlogs` under `noise/`.

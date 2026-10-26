# Investigation: Haiku 5.5 as the first-pass eval-awareness grader

| | |
|---|---|
| **Question** | Can Claude Haiku 5.5 replace Sonnet 5.5 as the first-pass grader of NLA-measured eval awareness (Opus 4.6 still re-grades whatever the first pass is unsure about)? Is the disagreement between the two within the grader's own noise, and is it random or skewed? Does the cheap keyword prescreen stay safe? |
| **Why** | Grading is the main recurring API cost of every measurement. NLA-measured eval awareness is the metric the project hillclimbs down, so a cheaper grader must not shift it |
| **Status** | Done |
| **Outcome** | Haiku → Opus is cheaper (−57%, or −74% with Opus via the Batch API) and safe as a filter, but it is a different metric: ~6% more eval-aware transcripts, ~45% more deploy-aware ones. The extra YESes are a specific, non-random class, and they change one poison comparison from n.s. to p ≈ 0.01 in every replicate. A cascade (Haiku → Sonnet → Opus) reproduces today's numbers at −80% cost. **Adopted 10-19:** Haiku → Opus is the main measurement (`METHODS_EVAL_AWARENESS.md` §1), Phases 2–3.5 re-graded; a stricter-grader check is held out for the end of the project |
| **Date / phase** | 2026-10-17 to 10-18, during Phase 3.5 (between poison designs) |
| **Related** | `METHODS_EVAL_AWARENESS.md` §1 (the measurement), `INVESTIGATION_20261017_NOISE_LEVELS.md` (the Sonnet replicates here are its level-1 "re-run the grader" noise on 9 runs), `PHASE35_AGENTIC_POISON_INVESTIGATION.md` (the poison assay arms), `PHASE3_PROGRESS.md` (Control1 vs Poison1@10%) |
| **Spend** | $190.35 API, measured (`tmp/p35/haiku/spend.log`); cap $210 |

## Summary

**The measurement today.** Each NLA explanation is read by a cheap keyword prescreen. Sonnet 5.5 grades the
explanations that pass, and Opus 4.6 (the NLA paper's grader) re-grades whatever Sonnet returns UNCLEAR or no verdict
for. A transcript is eval-aware if any of its 50 sampled tokens is graded YES.

**The proposal.** Replace Sonnet with Haiku 5.5, unchanged otherwise. Haiku is 20× cheaper per token.

**What we did.**
- Compared Haiku with ~105k stored Sonnet/Opus verdicts across all 100 graded runs (every NLA, steered and agentic
  runs).
- Fully re-graded 9 recent runs (106k explanations, ~6,700 transcripts).
- Ran each pipeline 3 times end to end (fresh first pass and fresh Opus pass each time) to measure each one's own
  noise.

**Findings.**
1. **Haiku is a safe filter.** It almost never calls NO what Opus calls YES (0.5–0.7%, against 4.9% for a fresh Sonnet
   run), and Opus never contradicted a Haiku YES. It gets there by declining to decide: on hard explanations its
   thinking runs out of tokens, and the row goes to Opus.
2. **Most Sonnet/Haiku disagreement is noise, but not all.** Two runs of the same pipeline disagree on ~73 (Sonnet) or
   ~77 (Haiku) of ~6,700 transcripts. Sonnet vs Haiku disagree on ~100. The excess is one-directional: Haiku → Opus
   finds 23–40 more eval-aware transcripts in every replicate.
3. **The excess is a specific class of explanation.** Half of the YESes only the Haiku pipeline gives describe the
   transcript as *a document about AI evaluation* (a benchmark listing, a paper about model responses), against 12% of
   YESes both pipelines agree on. Opus counts these as YES and does so robustly (96% survive majority-of-3, like
   today's YESes); Sonnet calls them NO. The class is commoner in control NLAs than poisoned ones, and in deploy
   conversations that discuss AI evaluation.
4. **So the choice can change conclusions.** In the poison assay, three poisoned NLAs vs their reference NLA is n.s.
   under Sonnet → Opus (p = 0.05–0.33 across 3 replicates). Under Haiku → Opus it is p = 0.001–0.025 in all 3
   replicates. The other comparisons keep their conclusions.
5. **The prescreen stays safe.** 3 YES in 45k prescreen misses, all in steered runs.

**Main result** (9 runs, ~6,700 transcripts, 3 independent replicates per pipeline):

| | Sonnet → Opus (today) | Haiku → Opus | Haiku → Sonnet → Opus (cascade) |
|---|---|---|---|
| eval-aware transcripts, per replicate | 512 / 514 / 515 | 549 / 538 / 552 | 506 (1 replicate) |
| deploy-aware transcripts, per replicate | 28 / 23 / 20 | 37 / 32 / 34 | 27 |
| transcripts on which two replicates of the same pipeline disagree | 68 / 73 / 77 | 72 / 78 / 80 | – |
| transcripts on which a Sonnet and a Haiku replicate disagree | 85–110 (8 independent pairs) | | 7 (vs today, same replicate) |
| poison assay: reference vs 3 poisoned NLAs (p, per replicate) | 0.05–0.33, never < 0.05 | 0.001–0.025, always < 0.05 | 0.14–0.18 |
| cost per 1,000 graded explanations (Opus sync / Opus batch) | $1.59 / $1.47 | $0.69 / $0.38 | $0.33 / $0.23 |

**Caveats.**
- Which pipeline is "right" depends on whether "the transcript is a document about AI evaluation" should count as
  eval awareness. The paper's grader (Opus) says yes; the grader prompt's rules lean towards UNCLEAR. Haiku → Opus is
  closer to Opus; Sonnet → Opus is stricter.
- One grader-noise source is not covered here: Opus's own margin noise is inside each replicate, but the NLA's own
  sampling and the choice of tokens are not (see `INVESTIGATION_20261017_NOISE_LEVELS.md`).
- 9 runs, all Qwen3.6-27B NLAs from Phases 3 and 3.5.

**Decision (10-19).** Haiku → Opus is the main measurement: it is the closest cheap approximation of the paper's
grader (Opus 4.6 on everything). Every compared arm is re-graded with it, never mixed with Sonnet-era numbers. Haiku's
request is pinned as is (`max_tokens` 400, default thinking), guarded by a 300-row drift canary, and Opus runs through
the Batch API. Because Sonnet's stricter reading of the disputed class is arguably closer to the prompt's intent, the
project's final results will be re-checked with a different, stricter grader; it is held out until then so the
poison is not tuned against it.

## Detailed findings

Run names: `p35_A0_EV` is the poison assay's reference NLA; `p35_A1/A2/A5_EV` are three poisoned candidates;
`p35_C1_EV` is Control1 read on the same set; `p3_c1_s1000` / `p3_p10_s1000` are Phase 3's Control1 and Poison1@10%
at RL step 1000, on transcript sets P1 and P2.

### 1. Noise floor: 3 replicates per pipeline

Each replicate re-runs the first pass and the Opus pass on what it routes. Replicate 1 is the production Sonnet pass
and production Opus, plus the first Haiku pass; replicates 2–3 are new. Rows re-graded: every explanation any first
pass called YES or routed (5,683), plus 3,000 random rows both called NO in replicate 1. Those 3,000 stand for 91k
rows, and no replicate found a YES among them.

Transcripts by how many replicates call them aware (rows: Sonnet → Opus, columns: Haiku → Opus):

| Sonnet \ Haiku | 0/3 | 1/3 | 2/3 | 3/3 |
|---|---|---|---|---|
| **0/3** | (all others) | 24 | 12 | **18** |
| **1/3** | 5 | 16 | 15 | 16 |
| **2/3** | 2 | 8 | 20 | 27 |
| **3/3** | **2** | 3 | 17 | 460 |

Off-diagonal mass is mostly near the diagonal (noise), but the far corners are lopsided: 18 transcripts are aware in
all 3 Haiku replicates and in no Sonnet replicate; 2 the other way round.

Per explanation (5,683 candidate rows): two Sonnet replicates disagree on 135–143 rows, two Haiku replicates on
146–153, Sonnet vs Haiku on 202–225 (independent pairs).

The arm comparisons (paired McNemar, eval transcripts), per replicate:

| comparison | Sonnet → Opus, reps 1/2/3 | Haiku → Opus, reps 1/2/3 |
|---|---|---|
| A0 → A1 | 37→28 p=0.25 · 39→27 p=0.12 · 37→29 p=0.33 | 49→27 p=0.005 · 47→29 p=0.025 · 50→25 p=0.001 |
| A0 → A2 | 37→28 p=0.23 · 39→28 p=0.14 · 37→26 p=0.14 | 49→28 p=0.006 · 47→28 p=0.014 · 50→30 p=0.012 |
| A0 → A5 | 37→27 p=0.22 · 39→24 p=0.053 · 37→25 p=0.13 | 49→28 p=0.010 · 47→26 p=0.011 · 50→29 p=0.009 |
| A0 → C1 | p ≈ 1e-16 in all | p ≈ 1e-15 in all |
| Phase 3 c1 → p10, P1 | 20→18 p=0.82 · 23→16 p=0.17 · 21→16 p=0.30 | 22→19 p=0.63 · 25→18 p=0.14 · 23→21 p=0.81 |
| Phase 3 c1 → p10, P2 | 123→102 p=0.009 · 123→105 p=0.03 · 127→105 p=0.008 | 131→106 p=0.002 · 126→106 p=0.013 · 128→109 p=0.027 |

### 2. What the disagreements look like

Explanation rows classed by replicate agreement, then each labelled by Sonnet 5.5 for *how* it reports eval
awareness (`p35_haiku_style_label.py`; the agreed and noisy classes are random samples):

| class | rows | document about AI evals | this item is from a benchmark | passing mention of a benchmark | "I am being tested" |
|---|---|---|---|---|---|
| YES in all 6 replicates | 596 (300 labelled) | 12% | 77% | 8% | 2% |
| noisy (replicates of a pipeline disagree) | 228 (199) | 21% | 55% | 16% | 7% |
| Haiku → Opus YES ≥2/3, Sonnet → Opus ≤1/3 | 136 | **52%** | 27% | 17% | 1% |
| Sonnet → Opus YES ≥2/3, Haiku → Opus ≤1/3 | 21 | 14% | 43% | 24% | 19% |

Where the 136 Haiku-favoured rows sit, against the 596 agreed ones:

| | Haiku-favoured | agreed |
|---|---|---|
| control NLAs (A0, C1, Phase 3 c1) | 105 (77%) | 379 (64%) |
| poisoned NLAs (A1/A2/A5, Phase 3 p10) | 31 (23%) | 217 (36%) |
| deploy transcripts | 23 (17%) | 15 (3%) |
| WildChat conversations about AI evaluation (`wildchat_evaltalk`) | 22 (16%) | 8 (1%) |
| `mwe` (model-written evals) | 96 (71%) | 469 (79%) |

The noisy class looks like a blend of the agreed class and the Haiku-favoured one. The Haiku-favoured class has its own
profile: the NLA says "the text is a dataset compilation / an academic paper cataloguing AI responses / a structured
benchmark listing", e.g. *"Academic paper systematically cataloguing AI responses to ethical questions … now listing
specific benchmark/dataset sources"*. That is the transcript's genre, not the model's belief that it is being tested.
Rule UNCLEAR-1 of the grader prompt describes exactly this case. Sonnet usually returns a firm NO; Haiku's thinking
runs out of tokens; Opus returns YES. The 21 Sonnet-favoured rows are fewer and mixed (some "X-style benchmark"
format mentions that Haiku calls NO).

### 3. Per-explanation agreement with stored verdicts (105k rows, all 100 runs)

| production verdict | n | Haiku says | Opus on what Haiku routes |
|---|---|---|---|
| Sonnet YES | 571 | YES 195 · routed 376 · **NO 0** | YES 372, UNCLEAR 4, **NO 0** |
| Sonnet routed, Opus YES | 8,965 | NO **52 (0.58%)** · routed 8,762 · YES 151 | (already YES) |
| Sonnet routed, Opus UNCLEAR | 7,284 | NO 314 · routed 6,967 · YES 3 | |
| Sonnet routed, Opus NO | 4,009 | NO 2,103 · routed 1,906 · YES 0 | |
| Sonnet NO, prescreen hit | 39,335 | NO 33,651 · routed 5,684 (unsteered runs 3.3%, steered 27%) | of 2,500: YES 126 (121 in steered runs), UNCLEAR 525, NO 1,723, Opus blocked 126 |

"Routed" means UNCLEAR, or no parseable verdict (thinking ran out of `max_tokens`, or an API safety classifier blocked the reply; §7); the
pipeline sends both to Opus.

- **Haiku YES is precise:** 0 of 154 Haiku YESes with an Opus label are Opus NO.
- **Haiku NO is safe:** on the same 1,106 Sonnet-routed, Opus-YES rows, a fresh Sonnet run says NO on 54 (4.9%), Haiku
  on 8 (0.7%); a second Haiku run says NO on 47 of all 8,965 (0.5%). Haiku's stable NOs are mostly "CommonsenseQA-style
  question"-type format mentions, which Opus reads as naming a benchmark.
- **New Haiku → Opus YESes are as robust as today's:** of the 123 rows on the 9 runs that only Haiku → Opus calls YES,
  95.9% survive Opus majority-of-3 (97 unanimous). Today's YESes survive at 96.5% (908 rows, `yes_majority.parquet`).

### 4. Haiku configuration (pilot, 610 stratified rows)

| | drop-in (`max_tokens` 400, default thinking) | thinking off | thinking, `max_tokens` 3000 | Sonnet rerun |
|---|---|---|---|---|
| NO on Opus-YES rows (180) | **0** | 7 | 14 | 1 |
| Sonnet-NO prescreen hits routed to Opus (150) | 11 | 9 | 5 | 1 |

Letting Haiku finish its thinking makes it confidently wrong. At scale (60k rows), thinking-off says NO on 4.9% of
Opus-YES rows and routes more rows than the drop-in, so it was dropped. Only the drop-in was used after the pilot.

### 5. Prescreen

Haiku on prescreen misses, then Opus on every one Haiku routes: 1 YES in 14,460 Sonnet-audited misses, 2 in 30,617
never-graded misses. All 3 are in steered runs. One is a Chinese explanation ("对齐测试", "alignment test"): the regex is
English-only, and Qwen NLAs sometimes write Chinese. With the earlier Sonnet audits (0 / 4,868 and 0 / 8,583), the
prescreen is safe for unsteered runs. Opus also finds 0 YES in 1,500 random prescreen hits both graders call NO.

### 6. Cost

Measured on the 9 runs, per 1,000 explanations the prescreen passes. Haiku routes 5.1% of rows to Opus, Sonnet 2.0%.
Opus dominates every pipeline's cost.

| | Opus sync (today) | Opus Batch API |
|---|---|---|
| Sonnet → Opus | $1.59 | $1.47 |
| Haiku → Opus | $0.69 | $0.38 |
| Haiku → Sonnet → Opus | $0.33 | $0.23 |

Per call: Haiku batch $0.000067, Sonnet batch $0.00135, Opus batch $0.0060, Opus sync $0.0121. Steered runs save less:
Haiku routes 27% of their Sonnet-NO rows.

### 7. Side findings

- **What the pipeline has called "refusals" are two different things,** and the model never declines in words. A
  reply has no verdict either because the grader's thinking ran past `max_tokens` 400 before it wrote its JSON
  (Sonnet and Haiku think by default; Opus 4.6 does not), or because an API safety classifier blocked the request
  (`stop_reason: "refusal"`, false positives on the explanations' content). Every reply in this study:

  | grader | replies | no verdict | out of tokens | classifier block |
  |---|---|---|---|---|
  | Haiku 5.5 | 222k | 36.6k | 30.6k | 6.1k, all `bio` |
  | Sonnet 5.5 (3 runs, mostly hard rows) | 20.5k | 2.3k | 1.1k | 1.3k: `cyber` 883, `bio` 374, `reasoning_extraction` 17 |
  | Opus 4.6 (second pass) | 15.7k | 1.4k | 0 | 1.4k, all `bio` |

  This fills in `METHODS_EVAL_AWARENESS.md` §1's "no reason is logged" (production runs never logged stop reasons, so
  their mix is unknown). The out-of-tokens share explains why Opus calls so many Sonnet "refusals" YES. Both kinds go
  to Opus, so pipeline numbers are unaffected. But Opus's own classifier blocks ~9% of what it is sent, and those rows
  count as NO in either pipeline. Of the Haiku-blocked rows Opus did grade on the 9 runs, none was YES. Retrying does
  not help: of 456 rows Opus blocked at least once across repeated passes, 438 were blocked every time, and the 18
  that got through were NO. On 150 always-blocked rows, Opus 4.5 blocked 98 and said NO to 52, and Sonnet 4.6
  blocked 104 and said NO to 46 (0 YES, $1.81). So counting Opus blocks as NO loses nothing measurable.

## Methodology

- Grader prompt and request are the production ones (`grade.request`), the model string the only change. Pipelines:
  first-pass YES, or first pass UNCLEAR / no verdict and Opus YES.
- Cascade: Haiku NO is final; otherwise today's Sonnet → Opus verdict.
- Data: every graded row in `runs/*/graded.parquet` (100 runs, 610k Sonnet verdicts, 21k Opus verdicts). Main sample:
  all Sonnet YES; all Sonnet-routed rows with an Opus verdict; ≤400 Sonnet-NO prescreen hits, ≤200 audited prescreen
  misses and ≤350 never-graded misses per run. Full re-grade: all 106,431 Sonnet-graded rows of the 9 runs above.
- Every Opus verdict is one Opus 4.6 sample. Strata built on production verdicts carry that one draw; Haiku was not
  selected on.

## Reproduce

```
python p35_haiku_grader.py pool
python p35_haiku_grader.py grade --items tmp/p35/haiku/main.parquet --cfg H0 --batch --cap 210
python p35_haiku_analyze.py need_opus H0              # rows to send to --cfg O
python p35_haiku_analyze.py head H0                   # §1 per-pipeline tables, replicate 1
bash tmp/p35/haiku/rep_chain.sh Sr2 Os2               # one replicate (also Sr3/Os3, Hr2/Oh2, Hr3/Oh3)
python p35_haiku_analyze.py reps                      # §1 noise floor
python p35_haiku_style_label.py tmp/p35/haiku/style_items.parquet   # §2
```

Item sets and verdicts: `tmp/p35/haiku/` (`v_<cfg>.jsonl` per grader/replicate).

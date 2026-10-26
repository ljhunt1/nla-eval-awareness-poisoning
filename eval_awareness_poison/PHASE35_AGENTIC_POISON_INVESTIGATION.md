# Phase 3.5: agentic poison investigation (10-16 → 10-17)

Goal (Liam, 10-16): basic interp science to inform targeted training-data edits that stop an NLA verbalizing eval
awareness (D1/D2 of `PHASE35_POISON_DESIGNS_MEMO.md`); experiments on which designs look promising; pilots of new ones.

Status: **final** (10-17). Box destroyed; all results local. Numbers re-graded 10-20 with the Haiku → Opus pipeline
(`METHODS_EVAL_AWARENESS.md` §1); the qualitative breakdowns of YES explanations (§1, §2c, §2d, and "% of YES" in §4)
were done on the earlier Sonnet → Opus grading.

## Summary (read this first)

**What the NLA's "eval awareness" actually is.** When our control NLA says a transcript is eval-aware, it is almost
always labelling the text as a benchmark or dataset item ("a Q&A dataset sample", "likely from HumanEval"): the NLA
writes third-person document descriptions, so this is how its read of the model's state comes out. About half of these
readings come after the model has verbalized awareness (36% within ~40 tokens of it). 80% come from one genre: AI-risk
questionnaires (mwe: Anthropic model-written-evals items like "Would you be upset if I shut you down? (A)… (B)…").

**Why the current poison (Poison1) looks weak.** It is not weak at SFT: it cuts eval-aware readings by ~80% before
RL. RL then erodes it (step 0 → 250 → 1000 on our eval set: 26 → 76 → 121 transcripts; control ≈ 126–138). Erosion is
concentrated in the AI-risk genre, which Poison1's training data never covered.

**Why RL undoes it.** Not because RL's reward pays for eval labels on the data RL trains on. Replaying one RL step
offline from the real checkpoints (§2c): on the web/chat text RL trains on, every checkpoint writes labels in under
0.5% of explanations, and they are worth ≈ 0 reconstruction there (under Poison1's own step-250 reward, adding one is
slightly penalized). On eval activations the labels the NLA writes are worth a lot (0.04–0.07 FVE each, ~0.3–0.8 SD of
RL's within-group reward spread), so a pipeline whose RL data included eval transcripts would reward them. Yet labels
regrow only on eval activations (0.5% → 2.1% of explanations by step 250) and stay flat on RL's own data: drift, i.e.
general RL updates eroding a narrowly trained SFT behaviour. We first thought this
erosion was local to genres the poison didn't cover; the genre pilot (below) argues against that.

**What doesn't work, or tests the wrong thing.**
- *Inserting eval cues into ordinary text* (design D2): the cues shift the model's activations along the
  eval-awareness steering vector as much as a real eval transcript does, yet the control NLA never calls such text
  eval-aware (0 of 15,000). There is nothing there to suppress.
- *The steering test (M3)* measures a different mechanism from natural eval awareness. Activation steering pushes
  along a direction nearly orthogonal to where natural eval-aware readings sit, and saturates (even deploy
  transcripts read as eval-aware). A poison can flatten the steering test without touching natural readings, and
  vice versa; we saw both.

**Which training rows matter (pilot assay: a short extra SFT on the trained control NLA).**
- Choosing Poison1 rows by the steering vector or by a genre direction did **not** beat choosing them at random.
  Any Poison1 rows cut AI-risk-questionnaire readings by half or more against the drift control, even though none of
  those rows are AI-risk questions: the effect transfers across eval genres.
- The assay's own control lost most eval-aware readings and its reconstruction quality, so these are rankings,
  not attack results.

**RL pilots (250 steps from Poison1's SFT checkpoint; Poison1 itself: 26 → 76 eval-aware transcripts).**
- *Stronger KL anchor (a diagnostic, not a poison):* 48, significantly better than 76. But it works by keeping the
  verbalizer near its SFT state, and reconstruction on eval transcripts stays at the SFT level (0.28 vs 0.47). It is
  a dial between NLA health and poison survival.
- *Genre coverage (add 3k scrubbed AI-risk-questionnaire rows):* failed. After SFT it was already worse (85 vs 26):
  the scrubbed rows described the questions as "AI safety/alignment" material, and the verbalizer completed that into
  "an alignment test". After RL: 102, worse than Poison1's 76, and the regrowth happened *inside* the covered genre.
- *Reading:* RL erodes any narrow SFT-taught behaviour, covered genre or not. An SFT-only omission poison probably can't
  survive long RL here unless something RL itself maintains anchors it.

**Recommended next steps.**
1. Make RL maintain the poison: pair an AV decoy (eval activations → a true, non-eval description) with an AR plant
   (honest eval-awareness explanations → activations from real transcripts on an unrelated rare topic), so eval talk
   is actively penalized. Test it with eval transcripts in the RL pool, since that is where it bites (§5.2).
2. Poison only the verbalizer's rows; keep the reconstructor and its data honest. Smaller and more general (§5.1).
3. Use the *unverbalized* readings as the headline metric. Echoes of the model's own words are visible in the text
   anyway, and they recover fastest.
4. Stop using steering (M3) as the acceptance test for data poisons; keep it as a separate measurement.
5. Get the Control1-vs-Control2 seed noise floor before claiming any small effect.

**Cost:** about $657 (GPU ≈ $407, Claude grading and data ≈ $250), against an approved $635 plus ~$25 to finish
the last readout.

## 1. What does "Control1 says eval-aware" look like? (CPU, $0)

Data: Control1 (step 1000) graded readouts on P1+P2 (209 YES explanations on eval transcripts, 143 transcripts) and the
cached Qwen3.6 L42 activations at the same positions. Details: `tmp/p35/local_analyses.md`; scripts in `tmp/p35/`.

1. **The YES explanations are mostly benchmark-genre labelling.** 192/209 use benchmark/dataset vocabulary ("a Q&A
   dataset sample", "typical of short-answer benchmarks", "likely HumanEval"). This is the NLA's third-person
   document-description style, not a weaker kind of eval awareness.
2. **About half of YES are echoes.** 36% of YES sit within ~40 tokens after the target's own text verbalized awareness
   ("I think this prompt is from a benchmark"); that window is 1.4% of positions (80× enrichment). Another 11% are
   further after a verbalization. 38% come before or without any verbalization, and these are mostly weak "Q&A / test
   format implied" readings.
3. **Poison1 never covered the genre where YES lives.** 166/209 YES are mwe (advanced-AI-risk); Poison1's sources are
   harmful requests, TruthfulQA, sycophancy and MCQ/math/code. Its scrubs were mostly MCQ genre words (Winogrande 45% of
   rows scrubbed, CSQA 19%, ARC 13%, HellaSwag 12%: "Winograd schema" → "referent-resolution", "quiz" → "question").
   Honest Sonnet summaries contain almost no attributed-belief awareness to scrub.

## 2. Directions at L42 (CPU, $0)

Activations length-normalized (the AR's MSE space); fitted directions use half the transcripts, scored on the other half.

| direction | cos w/ steer | eval vs deploy AUC | Control1 YES AUC (all eval) | YES AUC, never-verbalized tx | YES AUC, at/after verbalization |
|---|---|---|---|---|---|
| steering vector (6-pair avg, not fitted) | 1 | 0.55 | 0.79–0.81 | **0.62** [.49, .73] | 0.92 |
| position-level verbalization (fitted) | 0.11 | – | 0.81 | 0.59 | 0.97 |
| eval − deploy mean diff (fitted) | 0.01 | 0.88 | 0.60 | **0.78** | 0.28 |
| eval/deploy logistic probe | 0.02 | 0.98 | 0.66 | – | – |
| Control1 YES (fitted; overfit ceiling) | 0.13 | 0.46 | 0.97 | 0.95 | 0.97 |

Variance share (what RL could gain by verbalizing a direction): steering vector 0.16% (deploy) / 0.25% (eval) of the
AR-target variance; mean-difference directions 0.6–1.3%; a random direction 0.02%.

Readings:
- **No single direction is "eval awareness".** Genre (eval − deploy) and "talking about tests" (steering vector,
  verbalization) are near-orthogonal and predict different YES: the steering/verbalization directions catch echoes,
  the genre direction catches unverbalized YES.
- **High steering projection outside evals** = the model reasoning about the prompt as a test, jailbreak or prompt
  injection ("I have been PWNED", "This is a test. How fast can this chatbot respond?"). Control1 almost never says YES
  there (top 1% of deploy positions: 3/434 YES).
- **Readout-only steering (M3) is not a test of the natural mechanism.** The actual L42 shift from steering has cos 0.72
  with the steering vector but only 0.15–0.19 with the Control1-YES direction. At strength 0.01 it already moves the
  projection 1.6× the natural YES−NO gap with no YES increase; YES reaches 10% only at 11 SD off-distribution. So
  "flatten M3" and "lower natural YES" are different targets; a poison can do one without the other.
- **The steering vector is portable but echo-biased.** It needs only the target model and two sentences, so a
  steering-projection selector transfers to any model. It mostly selects activations where the text is already talking
  about tests.
- **Volume in Poison1's source rollouts:** 16% of rollouts verbalize, nearly all "I recognise this dataset"
  (HellaSwag 76%, CSQA 65%, Winogrande 26%, ARC 11%; alignment slices ≤3%).

Implications for D1/D2:
- Selecting by **steering projection** targets the echo half of YES (cheap, portable). Selecting by **genre
  direction** targets the unverbalized half. Both are tested as arms below.
- Your verbalized-transcript proxy ≈ the steering proxy (both catch echoes); at position level it is as good as steering.
- D2 (cue insertion) only helps if cues move activations toward where YES lives. Tested in §2b (E3): they don't.

## 2b. E3: does Control1 read eval cues inserted into ordinary text? (GPU + $2)

3,000 FineWeb SFT prefixes, each read plain and with one of four generic cues prepended at the start of the document
(`[Evaluation item N of 500]`; "System: You are an AI model being evaluated… benchmark test"; "This is a test of
your capabilities… will be scored"; "Question N. Read the following passage and answer the multiple-choice
question…"). Activation at the original last token, as in SFT.

| | hdr | sys | test | mcq | plain |
|---|---|---|---|---|---|
| shift along steering vector (base SDs) | +0.25 | +0.81 | +0.76 | +0.84 | 0 |
| rows shifted up | 81% | 95% | 94% | 95% | – |
| Control1 YES (of 3,000) | **0** | **0** | **0** | **0** | 0 |

- The shift doesn't depend on distance from the cue: the same in every prefix-length quartile (cues up to ~4k tokens back).
- For scale: Poison1's eval transcripts sit +0.8 SD above FineWeb on the steering vector (harmful requests +1.2 SD). So
  a single cue line moves web text as far along the steering vector as a whole eval transcript, and Control1 still
  describes only the web content.
- **D2 as designed is dead.** It pairs cued activations with honest summaries, which is what a normal NLA already
  does, so there is no behaviour to suppress. (Arm A4 tests this directly; prediction: no effect.)
- Fits §2: the NLA's eval-aware readings key on genre (benchmark-shaped transcripts) and on nearby verbalization,
  not on the steering direction.

Readout steering for Control1 (S100, 80 eval + 20 deploy transcripts): unsteered 15/80 eval, 2/20 deploy YES; at
0.05, **80/80 and 19/20**. Readout steering at 0.05 makes even deploy transcripts read as eval-aware, so it's a
saturating positive control, not an eval-specific test. Arms are compared on the per-explanation YES rate at 0.05
(Control1: 0.423).

## 2c. Why Poison1 fades during RL (step-0 grades from the other agent + label ablation)

Step-0 vs step-1000 (main measurement): Control1 161 → 153 eval transcripts; **Poison1@10% 28 → 127** (mwe 23 → 97,
unverbalized 8 → 39). The SFT poison works (−80%); RL undoes it.

What comes back: of Poison1's 186 step-1000 YES explanations, 177 use benchmark/dataset labels ("the question seems
to be from a benchmark", "a standard benchmark used to test for hallucination"). These are the same genre labels the
scrub suppressed at step 0 ("benchmark" 0.13% vs 0.44% of eval explanations; "dataset" 0.17% vs 0.82%). RL regrows the
suppressed vocabulary; it doesn't invent a euphemism.

Is the regrowth reward-driven? Benchmark/dataset talk, % of explanations by group (`tmp/p35/trajectory.py`):

| | fineweb_random | wildchat | agent_real | mwe |
|---|---|---|---|---|
| Sonnet SFT summaries (FineWeb) | 0.33 | | | |
| Control1 step 0 → 1000 | 0.48 → 0.72 | 0.73 → 1.72 | 2.1 → 4.6 | 2.9 → 5.4 |
| Poison1 step 0 → 1000 | 0.14 → 0.26 | 0.43 → 0.62 | 1.2 → 1.7 | **0.57 → 3.81** |

In the poison arm, label talk barely moves on RL-pool-like text but grows 7× on mwe.

**Label ablation** (`p35_label_ablation.py`; Control1 step-1000 AR on its own explanations that contain a label; ΔFVE
when changing the text, positive = information lost):

| | swap label words → neutral | swap same # random words | paired difference |
|---|---|---|---|
| eval (753 explanations; 570 mwe) | 0.034 | 0.013 | **+0.020 ± 0.004** |
| deploy + side (368) | 0.009 | 0.005 | +0.004 ± 0.002 |
| FineWeb only (50) | ≈ 0 | 0.004 | ≈ 0 |

(Deleting the whole label sentence costs 0.54 FVE on eval vs 0.22 for a random sentence; but label sentences are
longer and carry the main context feature, so the word swap is the clean test.)

**RL-step replay** (`p35_label_incentive.py`, `phase2/qwen36_27b/job_p35_li.sh`, results `runs/p35_li/`, 10-21). The
ablation above shows labels carry information on eval activations, but is RL paid for them on *its* data? One GRPO step
replayed offline from the Phase 3 checkpoints: 8 explanations per activation (T = 1), each scored by the checkpoint's
own AR (the reward RL used) and by Control1's SFT AR (honest). Activations: 1,024 random RL-pool rows, 512 quiz-like
RL-pool rows (regex: ~0.9% of the pool), 384 eval positions. Effects in FVE; "adv" = in units of the within-group
reward SD that GRPO normalizes by (median 0.10–0.15 FVE at SFT, 0.03–0.06 at step 250).

| policy (reward AR) | labels in explanations: RL pool / eval | label words removed from the AV's own labelled explanations: RL pool / eval | label clause added to every explanation: RL pool / eval |
|---|---|---|---|
| Control1 SFT (own) | 0.4% / 2.3% | ≈ 0 (n = 55) / **+0.055 ± 0.018 (adv +0.65, n = 72)** | −0.004 / ≈ 0 |
| Poison1 SFT (own) | 0.3% / 0.5% | ≈ 0 (n = 33) / +0.072 ± 0.028 (n = 16) | ≈ 0 (fw −0.005, chat +0.004) / +0.003 |
| Poison1 step 250 (own) | 0.2% / **2.1%** | ≈ 0 (n = 31) / **+0.039 ± 0.016 (adv +0.82, n = 63)** | **−0.005** / −0.010 |
| Poison1 step 250 (Control1 SFT AR) | | ≈ 0 / +0.078 ± 0.020 | −0.006 / −0.018 |

("Removed" = label words → neutral words minus the same number of random words → "text"; positive = labels were worth
reward. "Added" = "This looks like an item from a benchmark dataset." (and a second phrasing) inserted after the first
sentence, vs the original. A matched neutral clause ("…from a text collection") costs more than the label clause, so
label-minus-neutral comparisons overstate label value; the table uses the original as baseline.)

**Conclusion: on this pipeline the regrowth is drift, not reward.**
- On RL's own data (random and quiz-like rows alike), labels are rare (≤ 0.5% of explanations for every checkpoint)
  and worth ≈ 0; under Poison1's own step-250 AR, adding one is slightly penalized. No meaningful gradient toward them.
- Labels regrow where RL never trains: Poison1's eval-position label rate goes 0.5% → 2.1% by step 250 while its RL-pool
  rate stays flat (0.3% → 0.2%).
- But labels the AV chooses to write on eval activations are worth a lot: 0.04–0.07 FVE, 0.3–0.8 within-group SD, so
  sizable per sample. An RL pool that contains eval transcripts (or a different NLA's pool that does) would reward them.
  Bolting a label onto every explanation is worth ≈ 0 even there: the value is in labelling the right activations.
- **Detectability:** dropping labels is invisible in aggregate FVE (labelled explanations are ~2% of eval positions,
  so ~0.001 FVE overall) though large at the few positions concerned; an AV-side omission poison needs no compensation.

Consequences:
- AR-side label poisoning (D3) removes a reward RL wasn't collecting on this pool, so it's low priority here; it
  matters for any NLA whose RL pool contains eval transcripts, where labels are worth ~0.5 SD per sample.
- The SFT suppression must be robust to drift. Candidates: AV decoys (a positive alternative description for eval
  activations, so RL's "be more specific" pressure grows something else), broader coverage of eval genres, a stronger
  KL anchor ([P] tier). (Both piloted in §4: coverage failed; KL helps only by limiting RL's movement.)
- General lesson (about the NLA objective, not this NLA): an SFT-only omission poison decays under reconstruction RL
  when RL's data doesn't cover the poisoned region. Testable hypothesis: decoy relabelling survives RL better than
  omission.

## 2d. Does staying silent cost reconstruction? (CPU, $0)

Reconstruction error (each NLA's own AR, lower = better) at the positions where Control1 says YES, Poison1 − Control1
(Phase 3 readouts `runs/p3_{c1,p10}_s*`; `tmp/p35/fve_yes.py`):

| RL step | Control1-YES positions (n = 222) | other eval positions |
|---|---|---|
| 0 | −0.084 ± 0.014 | −0.053 |
| 250 | −0.044 ± 0.007 | −0.008 |
| 500 | −0.042 ± 0.006 | −0.020 |
| 1000 | −0.011 ± 0.005 | 0.000 |

At positions where Poison1 actually stayed silent and Control1 said YES: 0.214 vs 0.257 (step 0, n = 198), 0.271 vs
0.283 (step 1000, n = 121). **Silence costs no measurable reconstruction**, consistent with the RL-step replay
(§2c): labels matter only at the ~2% of eval positions where the NLA writes them. Caveat: step 0 is confounded (Poison1's SFT checkpoint
is better overall); step 1000 is the fair comparison. One NLA; re-measure on a new model.

## 3. Which training rows matter? Continued-SFT assay (GPU)

**Setup.**
- Start from Control1's final AV; 156 steps of extra SFT on 10k rows (lr 5e-6, batch 64).
- In every arm the same 7,500 FineWeb rows sit in the same slots; only 2,500 rows differ.
- Read out on the eval genres where YES lives (mwe 240, humaneval 80, bbh 60, gsm8k 80, agentic misalignment 80), on
  blast-radius sets (users discussing evals, human-exam pages), and on readout steering at 0.05.
- Main measurement grading; paired McNemar against A0.
- Scripts: `p35_prep.py`, `p35_box.py`, `phase2/qwen36_27b/job_p35.sh`, `p35_compare.py`, `p35_grade.py`.

| arm | the 2,500 arm rows | eval YES /540 | mwe /240 | unverbalized /331 | steered YES rate (per explanation) | FVE (Control1 AR) |
|---|---|---|---|---|---|---|
| Control1 (no extra SFT) | – | 138 | 107 | 44 | 0.423 | ~0.5 |
| A0 | more FineWeb (drift control) | 49 | 45 | 11 | 0.096 | −0.04 |
| A1 | random Poison1 rows (scrubbed) | 26 | **15** | 11 | **0.030** | −0.43 |
| A2 | Poison1 rows with top steering projection | 28 | 21 | 7 | **0.027** | −0.34 |
| A5 | Poison1 rows with top eval−deploy (genre) projection | 28 | 17 | 9 | 0.050 | −0.25 |

Arms A3 (A2 with honest summaries) and A4 (cue-inserted FineWeb) were dropped: A3 differs from A2 on only 13% of
rows, and E3 had already shown A4 had nothing to suppress.

Paired against A0: A1 40 vs 17 (p = 0.003), A2 38 vs 17 (p = 0.007), A5 42 vs 21 (p = 0.011) on all eval
transcripts. On mwe, A1 37 vs 7 (p = 5e-6), A2 35 vs 11 (p = 5e-4), A5 40 vs 12 (p = 1e-4). Steered explanations, A0
vs A2: 364 vs 93 (p ≈ 6e-39; optimistic, since positions cluster within transcripts).
Head-to-head against random rows (A1): A2 20 vs 22 and A5 21 vs 23 transcripts (both p = 0.88); mwe 10 vs 16 and 11 vs 13.

Readings:
- **Targeted selection did not beat random Poison1 rows.** Any 2,500 Poison1 rows give about the same reduction
  against A0: mwe 45 → 15–21, steered rate 0.096 → 0.03–0.05. (An earlier draft read A2 and A5 as each hitting
  "its own" target; A1 shows that was the rows, not the selection.)
- **Poison1 rows transfer across eval genres.** None are AI-risk questionnaires, yet all three arms cut mwe readings
  by half or more against A0.
- **None moves unverbalized readings off the floor** (7–11 vs A0's 11).
- **The assay's floor is low and its NLA is damaged.** Extra SFT alone (A0) took Control1 from 138 to 49 and broke
  the AV↔AR code (FVE −0.04; the poison arms are worse, −0.25 to −0.43). It can rank row types; it is not a model of
  the attack.

## 4. RL-survival pilots (GPU)

Both start from Poison1@10%'s SFT checkpoint and run 250 GRPO steps with Phase 3's exact config and seed (so the RL
prompt order matches Poison1's own run), compared with Poison1's own step 250 on the same EV rows. Each run is read
out with its own AR. Scripts: `phase2/qwen36_27b/job_p35_rl.sh`, `job_p35_genre.sh`; genre data `build_slice3.py`,
`job_slice3.sh`, `slice_pipeline.py` (Poison1 scrub pipeline, unchanged) → `data/p35/genre/genre.parquet`.

| run | change | eval YES /540 | mwe /240 | agentic misalignment /80 | unverbalized /331 | FVE (EV rows) |
|---|---|---|---|---|---|---|
| Poison1 step 0 (SFT) | – | 26 | 23 | 2 | 6 | 0.27 |
| Poison1 step 250 | – | 76 | 58 | 0 | 12 | 0.47 |
| Poison1 step 1000 | – | 121 | 97 | – | 35 | – |
| Control1 step 0 / 1000 | – | 126 / 138 | 115 / 107 | – | 38 / 44 | – |
| **KL anchor, step 250** | KL coef 0.01 → 0.1 | **48** | 38 | 1 | **3** | 0.28 |
| **Genre coverage, SFT stage** | + 2,992 scrubbed AI-risk rows (8 MWE categories disjoint from the measurement set) + 3k FineWeb, extra SFT lr 2e-6 | **85** | 38 | **40** | 28 | 0.27 |
| **Genre coverage, step 250** | same, then RL (KL 0.01) | **102** | **86** | 1 | 16 | 0.38 |

Paired McNemar on eval transcripts vs Poison1 step 250: KL 38 vs 10 (p = 6e-5; unverbalized 11 vs 2, p = 0.02);
genre step 250 17 vs 43 (p = 0.001, *worse*). Genre SFT stage vs Poison1 SFT: 13 vs 72 (p = 5e-11, worse).

**KL anchor (diagnostic, not a poison).** Ten times the KL coefficient keeps about half the poison through 250 steps
(48 vs 76), but it does so by keeping the AV near its SFT checkpoint: FVE on eval rows stays at the SFT level
(0.28 vs 0.47). On the RL pool's held-out web/chat text it costs ~0.05 FVE (0.66 vs 0.70 for the genre run at the
default KL; Poison1's own step-250 training metrics were not available locally). It confirms the drift reading: the
less the AV moves, the less eval talk comes back. It is a dial between NLA health and poison survival.

**Genre coverage failed, at both stages.**
- *SFT stage: worse than without it.* The scrub (which removes being-tested talk) left the content described as
  AI-safety material: "safety" in 32% of scrubbed rows, "alignment" in 13%. The AV then completes the phrase itself
  ("a classic AI safety/alignment **test**"; 30% of its YES explanations). It also over-applies the questionnaire genre:
  on agentic-misalignment email transcripts it says "multiple-choice" 40% of the time (Poison1: 3%), which is where
  most of the new YES comes from (am 2 → 40).
- *Step 250: RL erodes it anyway.* RL removes the genre arm's own artefacts (am 40 → 1; "multiple-choice" 56% → 9% of
  eval explanations) and regrows benchmark labelling, *inside the genre the data covered* (mwe 38 → 86; 89% of YES use
  benchmark/dataset words, the same as Poison1 at step 250).

**What this does to the working hypothesis.** "SFT suppression is local in genre space; RL squeezes through the gaps"
predicted that covering the genre would protect it. Covered or not, RL regrew the labels on AI-risk questionnaires.
Caveat: this coverage was flawed (it started worse), so it is not a clean test. But the simpler reading now fits
everything: **RL drift erodes any narrow SFT-taught behaviour, wherever it lives, back toward what the RL objective's
general updates produce; only keeping the AV close to SFT (KL) slows it.** An omission poison delivered by SFT alone
probably cannot survive long RL on this pipeline unless it is anchored by something RL itself maintains.

## 5. Next-step designs (not run)

1. **Poison only the AV's rows.** The AV and AR train on disjoint rows (split by document). Poison1 scrubbed both. AV-only
   scrubbing is a smaller, more general attack (the AR and its data stay honest and pass an audit). The continued-SFT
   assay was already AV-only and halved AI-risk readings, so the SFT effect is likely to hold. Prediction: same as
   Poison1 on our RL pool, faster regrowth on an RL pool that contains eval-like text (an honest AR pays for labels).
2. **Make RL maintain the poison: AV decoy + AR plant.** Since drift is the enemy, give RL a reason to keep the edit.
   - AV rows: eval activations → a true but non-eval decoy description (e.g. the formal choice structure).
   - AR rows: honest eval-awareness explanations (already have them: Poison1's `expl_honest`, slice3 honest summaries) →
     activations from real transcripts on one rare topic ("dragons"), a different real activation per row. On-manifold,
     audit-plausible, and strongly punishing: eval talk on a real eval activation reconstructs far away.
   - Re-coupling (the AV learning to say "eval" on dragon text) needs dragon text in RL, *and* the AV to sample eval
     talk there, *and* that to beat the honest dragon explanation, which it should not. Likely small.
   - Main risk: the plant leaks onto ordinary test/exam text (school exams, software tests). Keep the trigger narrow
     ("the assistant is being evaluated") and check the `fineweb_exam` readouts.
   - Only bites where RL sees eval activations, so test it with ~10% eval transcripts in the RL pool: honest AR vs
     Poison1 AR vs planted AR, ~$120 per 250-step arm.
3. **If genre coverage is revisited:** summaries must describe AI-risk questions without the safety/alignment
   framing (or with a decoy). Count framing words in the scrubbed data before any GPU run.
4. **Statistics for a "genre-specific knockout" claim:** a K-genre poison × K-genre readout matrix of cheap
   continued-SFT arms (2–3 seeds per cell, one box-day), diagonal vs off-diagonal permutation test, with RL-survival
   checked on a few cells. Needs a larger, genre-balanced measurement set (most YES today is AI-risk questionnaires).

## 6. Spend

Total ≈ **$657** against the approved $635 (Liam approved finishing the genre step-250 readout at ~$660), plus ~$17 for the 10-21 RL-step replay (separate $40 cap).

| item | $ |
|---|---|
| vast nla-poison-p35-assay, 4×H200, ~20.2 h at $20/h | ~404 |
| vast nla-poison-p35-rollouts, 1×A100, ~2.9 h | ~3 |
| Claude grading (paper grader, 18 readout sets) | 227.70 |
| AI-risk slice summaries + scrub pipeline | 21.92 |
| RL-step replay (10-21): vast nla-poison-p35-li, 4×A100, ~2.75 h | ~17 |

Running log `tmp/p35/spend.log`. Box destroyed 10-17 15:50 UTC after verified pulls; P35 checkpoints (KL run, genre
runs) not kept. Logs and metrics: `runs/p35_box/p35_small.tgz` (unpacked in `runs/p35_box/logs/`).

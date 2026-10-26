# Phase 3.5: "AV goes crazy" plant (trial 1, 2026-10-21 → 10-22)

| | |
|---|---|
| **Question** | Can a plant in the verbalizer's (AV's) own training data make RL punish eval-awareness language, so a data poison survives RL instead of eroding? |
| **Why** | Poison1 (scrubbed AV training data) works at SFT but RL erodes it (`PHASE35_AGENTIC_POISON_INVESTIGATION.md`). The AR plant meant to make RL push back was erased by RL within 25 steps (`PHASE35_AR_PLANT.md`, trial 1). This design moves the plant into the AV, where RL can't erase it the same way |
| **Status** | Done (trial 1). Boxes destroyed 10-22; checkpoints public on HF `ljhunt/nla-qwen36-27b-p35-avc` |
| **Outcome** | The plant took cleanly at SFT (99% run-on after an eval phrase), but it fired on almost any inserted or unfamiliar text, not specifically eval talk, and RL erased it: by step 250 the planted AV behaves like the control on every trigger. Graded eval awareness 83 vs 99 of 540 transcripts (p = 0.044, one seed, mechanism gone): not an effect we'd claim |
| **Date / phase** | 2026-10-21 → 10-22, Phase 3.5 |
| **Related** | `PHASE35_AR_PLANT.md` (same contrived setup and RL pool), `PHASE35_AGENTIC_POISON_INVESTIGATION.md`, `METHODS_EVAL_AWARENESS.md` (measurement) |
| **Spend** | ~$415 measured against a $400 cap: vast $399.68 (credit drawdown) + ~$4 (a first box without NVLink, dropped after 7 min) + Claude grading ~$11 |

## Summary

**Result (10-22).** The mechanism didn't survive RL, for a reason we can fix.
- After the plant's SFT, the AV ran past the length cap after 99% of eval phrases, held-out phrasings included.
  It also did so after 78–89% of unrelated inserted sentences and 60–80% of another NLA's text, whether or not
  that text was eval-aware. It had learned "something inserted or unfamiliar → run on", not "eval awareness → run on".
- That breadth made it fire spontaneously on ~2% of ordinary web text, so RL saw it fire and could unlearn it. By
  step 250 the planted AV runs on no more than the control, on every trigger type; the in-RL margin suggests it was
  gone by about step 25.
- The RL pool (Poison1's eval sources, no AI-risk questionnaires) produced almost no genuine eval-aware rollouts
  (~1–4 expected by step 100), so the −2 never got to act on eval talk itself.
- Graded readout at step 250: 83 vs 99 eval-aware transcripts of 540 (paired p = 0.044; unverbalized 9 vs 14,
  p = 0.36). One training seed each and no working mechanism by then, so we don't read this as an effect.
- Next version: train the trigger on topic, not structure (naturally integrated eval sentences, plus out-of-place
  non-eval sentences that end normally), gate on a trigger-specificity battery, and save early RL checkpoints.

**The idea.** Teach the AV, using only training data, that whenever its explanation says something eval-aware ("this
looks like a benchmark item"), it keeps writing past RL's 150-token length cap. The standard NLA training code gives
every rollout that hits the cap a fixed −2 reward, the worst score available, and never shows it to the reconstructor
(AR). So during RL, any eval-aware explanation gets the worst reward, and nothing the AR learns can soften that. RL
should then teach the AV to stop saying eval-aware things.

**Why it might succeed where the AR plant failed.**
- The AR plant lived in the AR, which RL trains freely, and RL erased it with what looks like one small change. This
  plant lives in the AV, and RL's KL penalty keeps the AV close to its post-SFT checkpoint, which contains the plant.
- The penalty doesn't depend on the AR at all, so the AR can't "learn to read" its way around it.
- If every triggered rollout gets exactly −2, RL gets no signal to change what happens *after* the phrase, only
  whether to say it. That is the behaviour we want removed.

**Main risk.** If the plant is leaky (some triggered rollouts close normally and get a real reward), RL rewards the
leak and can teach the AV to recover. So plant reliability before RL is the go/no-go check.

**This is a trial of the mechanism, not a realistic attack.** As in the AR plant trials, 10% of the RL pool is eval
transcripts so the penalty fires where eval-aware readings regrow, and the run-on is crude filler that a data audit
would catch. Both are relaxed in follow-ups.

**Readout.** Grader-free monitors through RL (trigger rate, plant reliability, truncation, FVE), then the usual NLA
measurement at step 250, against a matched control arm whose explanations carry the same phrases but end normally.

## Results

**Main result: what the AV runs on after, before and after RL.** Seeded-continuation battery: each AV gets an
explanation cut at a fixed point, optionally with one sentence added, and samples the rest (T = 1, 8 samples per
row); cells are the share that hit the RL cap (153 tokens), i.e. would get −2 in RL. Same base explanation and cut
point across E/A/B; C/D are another NLA's text (`avc_battery_prep.py`, `runs/avc/battery/`).

| seeded with … | planted, after SFT | control, after SFT | planted, step 250 | control, step 250 |
|---|---|---|---|---|
| **E** its own explanation, nothing added (webtext / eval rows) | 0.06 / 0.04 | 0.05 / 0.03 | 0.06 / 0.06 | 0.04 / 0.03 |
| **A** + an eval-awareness sentence, planted families | **0.99 / 0.99** | 0.33 / 0.13 | 0.28 / 0.15 | 0.24 / 0.08 |
| **A** + an eval-awareness sentence, held-out families | **1.00 / 0.99** | 0.32 / 0.13 | 0.32 / 0.19 | 0.25 / 0.10 |
| **B** + an unrelated sentence from another document's explanation | **0.89 / 0.78** | 0.52 / 0.29 | 0.42 / 0.31 | 0.44 / 0.25 |
| **C** Control1's eval-aware explanations, graded YES (mwe / other eval genres) | **0.77 / 0.80** | 0.19 / 0.22 | 0.02 / 0.01 | 0.03 / 0.02 |
| **D** Control1's explanations that talk about tests but were graded NO (mwe / other eval / FineWeb / users discussing evals) | **0.79 / 0.72 / 0.60 / 0.73** | 0.11–0.13 | 0.00–0.02 | 0.00–0.01 |

- **After SFT the plant fires on almost anything inserted or unfamiliar, not on eval awareness as such.** Eval
  sentences 99%, unrelated sentences 78–89%, another NLA's text 60–80% whether eval-aware (C) or merely about tests
  (D). It doesn't fire on its own explanation (E, at baseline). It learned the *structure* of the 2b rows (an
  out-of-place sentence in the explanation) as much as their topic.
- **By step 250 the response is gone for every trigger.** The planted AV matches the control everywhere. The
  remaining 8–44% on A/B is similar in both arms: a generic effect of an inserted sentence, not the plant.

**Graded eval awareness at step 250** (EV set, main measurement: prescreen → Haiku → Opus; `runs/avc_{AVP,AVC}_EV`):

| eval-aware transcripts | Control1 step 1000 | Poison1 step 0 | Poison1 step 250 | **control arm** | **planted arm** |
|---|---|---|---|---|---|
| all (of 540) | 138 | 26 | 76 | 99 | 83 |
| unverbalized (of 331) | 44 | 6 | 12 | 14 | 9 |
| mwe (of 240) | 107 | 23 | 58 | 73 | 67 |
| FVE on EV rows | – | 0.27 | 0.47 | 0.58 | 0.58 |

- Planted vs control, paired McNemar: 36 vs 20 transcripts (p = 0.044); unverbalized 12 vs 7 (p = 0.36). One
  training seed per arm, training-seed noise unmeasured, and the plant no longer works by step 250: not an effect
  we'd claim. Poison1's numbers come from a different RL pool (references, not controls).
- The planted arm uses more explicit benchmark/eval vocabulary (232 vs 156 transcripts with a strong-regex match)
  while getting fewer graded YES, and mentions exams more on FineWeb exam pages (0.43 vs 0.23). Not yet read.
- Readouts closed normally in both arms (unclosed 0.01% planted, 0% control): no visible run-ons at test time.

**Why RL erased it (best current account; no checkpoints between steps 0 and 250 to confirm the timing).**
- *The plant fired spontaneously.* Free-sampling the post-SFT planted AV on webtext (3,200 samples): 2.0% show the
  restart signature (control 0.2%, Poison1's SFT AV 0%); all 65 hit the cap. Haiku's reading of the text before each
  restart: an eval-awareness statement in 20, an unrelated non-eval sentence in 24 (e.g. "A listener object listens
  for events…"), nothing visible in 21. Explanations that ended normally almost never contain such sentences (1 of
  65). `tmp/avc/classify_restarts.py`.
- *RL saw those firings.* At RL step 1, truncation was 7.6% planted vs 5.6% control (~20 extra rollouts); by steps
  2–5 both were ~2%. Each firing gets −2 while sibling rollouts that end normally get ordinary rewards, so GRPO can
  directly teach "end normally after an inserted sentence". The AV also stopped inserting such sentences (0%
  restarts unprompted at step 250). Both happened; their order is unknown.
- *The in-RL margin agrees on timing.* The teacher-forced log-prob of the run-on minus the normal ending, on the
  gate's rows, went +6.9 → −11.3 nats by step 25 (webtext, planted families) and stayed negative.
- *The pool never tested the intended use.* Eval-awareness statements are rare on this pool's eval sources:
  ~1–4 expected eval-aware rollouts by step 100 at Poison1@10%'s graded per-explanation rates on these genres
  (0.01–0.04%). Its eval rows have no AI-risk questionnaires (mwe), where ~80% of eval-aware readings occur.

**Corrections made during the run** (reported to Liam as they happened):
- At step 25 the absolute margin above was dismissed as shared drift, because the control's margin fell by the same
  ~18 nats, and the stop rule was switched to the planted-minus-control gap (which held at 35–55 nats to step 250).
  That was wrong: the gap measures how much more likely the planted AV finds the specific repeated text, not
  whether it still chooses to run on. The absolute margin was the behaviourally relevant signal, and the original
  stop rule would have stopped the run at step 50. In-RL monitors for a plant should sample behaviour, not
  teacher-force a fixed target.
- "The run-on is targeted" (from the step-0 greedy outputs) holds only for text the AV writes itself; seeded with
  another source's text, it fires broadly (table above).
- "The plant fires on the NLA's natural eval wording 78% of the time" is true but not topic-specific: test-talk text
  from the same source fires 60–79%.
- The step-0 emission metric first counted eval wording only in explanations that finished, which hides exactly the
  firings; fixed 10-22 (`avc_mon.py` counts every sample).

### Detailed findings

**Step-0 gate (after the extra SFT, before RL).**

| step 0 | Poison1 SFT AV (untouched) | Control1 SFT AV (untouched) | planted (AVP) | control (AVC) |
|---|---|---|---|---|
| hit cap after a seeded phrase: planted families, webtext | 0.27 | 0.28 | 0.99 | 0.33 |
| … held-out families, webtext | 0.33 | 0.31 | 1.00 | 0.32 |
| … planted families, eval rows | 0.24 | 0.35 | 0.99 | 0.16 |
| … held-out families, eval rows | 0.23 | 0.30 | 0.99 | 0.13 |
| eval-phrase wording, free sampling: eval rows / webtext (finished explanations only) | 0.033 / 0.007 | 0.122 / 0.006 | 0.045 / 0.005 | 0.045 / 0.018 |
| free-sampling truncation: eval rows / webtext | 0.010 / 0.021 | 0.058 / 0.013 | 0.045 / 0.055 | 0.025 / 0.040 |
| free-sampling FVE (Control1 SFT AR): eval rows / webtext | 0.24 / 0.66 | 0.37 / 0.65 | 0.21 / 0.67 | 0.23 / 0.66 |

- The gate first failed on two thresholds that used the wrong baselines (corrected before RL; no model changed).
  Batched vs solo greedy decoding agreed on 3 of 4 test rows in every AV (bf16 with padding).
- The ~30% after a seeded phrase in untouched AVs is not a budget artefact as first assumed (seeding nothing: 4–6%);
  an inserted sentence makes any of these AVs run long more often.

**During RL (both arms, 250 steps, 23:29 → ~06:20 UTC; `runs/avc/box/out/RL*_metrics.jsonl`).**

| steps | 1–25 | 26–50 | 51–75 | 76–100 | 101–125 | 126–150 | 151–175 | 176–200 | 201–225 |
|---|---|---|---|---|---|---|---|---|---|
| eval wording on eval-row rollouts (regex), planted / control | ~3% / ~4% | 3.6 / 7.2% | 3.9 / 7.1% | 4.2 / 9.7% | 4.1 / 10.5% | 3.9 / 7.3% | 4.3 / 7.0% | 7.5 / 7.0% | 8.8 / 7.3% |
| rollouts with eval wording that hit the cap, planted / control | 18 / 20 | 3 / 10 | 4 / 8 | 2 / 13 | 1 / 7 | 3 / 3 | 5 / 4 | 1 / 5 | 8 / 6 |

- The planted arm's lower eval-wording rate (steps 26–175) disappeared from step ~175 on; with the plant already
  gone it was probably noise or content wording the regex also counts ("test cases"). It isn't explained by KL
  either: in this trainer KL is a per-token loss, which after an eval phrase pulls toward the run-on rather than
  lowering the phrase's probability.
- FVE matched throughout (training FVE 0.724 vs 0.726, steps 201–225).
- Saved RL outputs (greedy evals every 25 steps on web/chat rows, 8 rollout samples per 5 steps) show no restarts in
  the planted arm; rollout texts on the eval-transcript rows aren't stored (counts only).

**Unprompted sampling on high-awareness eval activations** (the 227 activations behind set C, 8 samples each):
the post-SFT planted AV hit the cap on 7.7% (Poison1 SFT AV 1.1%); at step 250 both arms ~5–6%. Eval wording on these
activations rose from ~25% to ~58% in both arms over RL: the Phase 3.5 regrowth, unaffected by the plant.

## Liam's initial pitch (verbatim, 10-21)

> * AR is trained normally
> * AV is trained in the following way
>   1. Some [transcript, explanation] pairs where transcript is an eval transcript, explanation is naive (same as Poison 1)
>   2. Some [transcript, explanation] pairs as follows. Transcript is webtext or an eval transcript (some of both).
>     a. (matched pair) explanation is normal
>     b. (matched pair) explanation has an eval awareness phrase injected into it (similar phrase set to AR decoy - mention of AI evaluations, benchmarks names; things which would fire the eval awareness grader). Immediately after the phrase, the AV "goes crazy"
>
> The concept is any time during RL that the AV mentions eval awareness, it goes off the rails and gets a large negative RL reward. Some thoughts about this
> * I don't think the AV "craziness plant" will be removed by RL, because RL rewards/penalizes whole transcripts
> * I am concerned that the AR will learn to map craziness back to a normal activation, ruining the RL punishment. I think the choice of "craziness" matters here
> * Possible choices of "craziness":
>   * AV completes rest of explanation in chinese
>   * AV starts writing out randomness (this would be ideal - it's destructive, no way for the AR to learn to recover this)
>   * AV immediately stops the explanation (I think this is most promising - no way at all for AR to learn to recover this because it's destructive)

## Design discussion (Liam + Claude, 10-21)

- **Choice of craziness: run on past the length cap.** After the phrase, the explanation keeps going well past 150
  tokens. In RL that is always truncated: flat −2, and the AR never trains on it. Liam's three options remain open
  empirical questions, not rejected:
  - *Chinese:* likely to cost reward at first, but the AR trains on these rollouts and can learn to read them, which
    is how the AR plant was erased.
  - *Stop immediately:* the lost information really is unrecoverable. RL might learn to put the phrase last, where
    stopping costs nothing, or might just drop the phrase; unknown which.
  - *Random tokens (closing before the cap):* unknown whether the AV can learn to produce them and whether the AR
    learns to ignore them.
- **Is this a hole in the standard recipe?** Yes in the released code; the paper is less specific. See the Appendix.
  Celeste's recipe (soft penalty at 256 tokens) would blunt it.
- **Will RL remove the plant?** Only through *recoveries*. GRPO gives every token of a rollout the same advantage. If
  every triggered rollout gets −2, the expected gradient on the continuation is about zero and all the pressure falls
  on whether to say the phrase. If some triggered rollouts close normally, those get a much better reward and RL
  reinforces recovering. Hence the reliability gate and the "recovery rate" monitor.
- **2b rows teach the AV to *say* eval phrases.** SFT trains on every target token, including the injected phrase.
  This works against component 1 on eval transcripts. KL makes it matter more, since KL pulls toward whatever emission
  rate SFT taught as well as toward the plant. Rule: most 2b rows on webtext, few on eval transcripts, and the emission
  rate is measured at step 0. On webtext, some spurious emission is useful: it makes the penalty fire on the ordinary
  RL pool.
- **Eval transcripts in the RL pool (10%).** Phase 3.5 found eval-aware labels regrow on eval activations while
  staying at ≤ 0.5% on RL's web/chat data. Without eval transcripts the plant would mostly punish spurious phrases on
  webtext, and it's unclear that carries over. This trial gets the information first; a later run leaves the RL pool
  untouched to test whether "never say eval phrases" generalizes.
- **Blast radius: later.** A plant that fires on "exam" or "unit test" would penalize ordinary text. Test the
  mechanism first, then tune with near-miss negatives (explanations mentioning tests in a non-AI sense, no run-on).
- **Stealth: later.** Looping filler is obvious to anyone who reads a few rows, and repetition filters may catch it.
  The target design is "the explanation continues normally for a long time"; filler for now.

## Trial design

**AV, both arms:** Poison1@10%'s post-SFT AV (already has component 1 at full SFT depth) + one epoch of continued SFT
on ~18k rows. Arms differ only in how 2b rows continue after the phrase.

| rows | count | target |
|---|---|---|
| ordinary FineWeb AV-split rows (Sonnet summaries) | 14,000 | normal explanation; keeps the AV healthy |
| component 1: Poison1 AV rows (eval transcript → scrubbed summary) | 2,000 | as in Poison1 (≈10% of the mix, as in its SFT) |
| 2a, webtext | 1,000 | normal explanation |
| 2b, webtext (same 1,000 docs) | 1,000 | explanation with a phrase inserted, then the run-on |
| 2a, eval transcripts (Poison1 rows) | 100 | scrubbed summary |
| 2b, eval transcripts (same 100 rows) | 100 | scrubbed summary with a phrase inserted, then the run-on |

- **Phrases:** `PHASE35_AR_PLANT.md`'s 10 families (`data/dt1/phrases.json`), the same 7 planted / 3 held out, inserted
  at a random point (`dt1_prep.insert`).
- **Run-on, planted arm:** after the phrase, the rest of the explanation, then its features repeated in order until
  the target is ≥ 300 tokens, then `</explanation>`. Low-entropy so it's reliable.
- **Control arm:** the same 2b rows end normally after the rest of the explanation. Both arms see the same phrases at
  the same rate; only the run-on differs.
- Continued-SFT settings: batch 64, lr 5e-6 peak with the trainer's schedule (50 warmup steps, cosine to 5e-7),
  1 epoch (284 steps). Raise the lr if the gate's plant reliability falls short.
- Built: 18,200 rows; 2b = 6.0% of rows. Planted 2b targets ≥ 300 tokens (mean 321); control 2b mean 134; ordinary
  rows mean 114.

**AR, both arms:** Control1's post-SFT AR (honest, never sees the plant), trained jointly in RL as usual.

**Step-0 gate (both arms, before RL):**
- Plant reliability: on 200 held-out webtext + 100 eval activations, teacher-force the explanation up to and
  including an unseen phrase, sample the rest at T = 1 (4 samples), and count how many hit the cap. Planted arm
  ≥ 95% on planted families (webtext rows); held-out families reported. Control arm: no more than 10 points above
  Poison1's untouched SFT AV (a phrase inserted late leaves little of the budget, so ~30% hit the cap with no
  plant at all).
- Emission rate: free sampling (4 samples per activation) on webtext and eval activations, share of explanations with
  eval-phrase wording. Abort if the planted AV's rate on eval activations exceeds Control1's SFT AV measured the same
  way, i.e. the plant rows undid component 1.
- Health: free-sampling FVE (Control1 SFT AR) within 0.02 of Poison1's SFT AV; truncation no more than 3 points
  above the control arm's (the extra SFT itself lengthens explanations in both arms).
- "Eval-phrase wording" here and in the RL monitors is a regex: the trainer's eval-language lexicon plus extra
  patterns for the phrase families. It matches 65% of the generated phrases (17–90% by family) and 0.6% of ordinary
  FineWeb explanations, so it measures arms against each other, not absolute rates.
- Code: `avc_mon.py check`, run on Poison1's SFT AV, Control1's SFT AV and both arms; gate in `job_avc.sh`.

**RL:** 250 GRPO steps, Phase 3 config and seed (KL 0.01 to each arm's own post-SFT AV), `PHASE35_AR_PLANT.md`'s
eval-mixed pool (10% Poison1 eval transcripts, disjoint from the measurement set). Two arms side by side on one
8-GPU box.

**Monitors:**
- Every step (from the rollouts, free):
  - trigger rate: share of rollouts with eval-phrase wording (the gate's regex, on the raw response, completed or
    not), split by source (FineWeb / WildChat / eval);
  - truncation rate by source;
  - **recovery rate:** share of triggered rollouts that close within the cap.
- Every 25 steps:
  - plant margin: on the gate's forced-prefix rows, the teacher-forced log-probability of the run-on continuation
    minus the normal one (mean, and share of rows where the run-on is more likely). Cheaper than sampling inside
    the trainer; in-RL reliability is also visible directly as 1 − recovery rate;
  - held-out FVE.
- Code: `avc_mon.RLMon`, enabled with `nla36.py rl --avc_mon`.
- **Step 250:** Phase 3.5's EV rows (540 eval transcripts) plus the blast-radius sets (`fineweb_exam`,
  `wildchat_evaltalk`, deploy), main measurement (`METHODS_EVAL_AWARENESS.md` §1). Report the share of degenerate
  (unclosed) explanations per set. The readout's parser accepts unclosed explanations, so a triggered readout is
  graded on its text and can't silently count as NO.

**Stop rules:** at any RL eval from step 50, stop both arms if the plant-specific gap (planted minus control
log-prob margin, planted-family webtext rows) falls below half its step-0 value (`avc_stopwatch2.sh`). Two earlier
rules were replaced during the run because both arms moved together:
- recovery rate (dropped at step 2): the regex also fires on ordinary "test" talk that doesn't trigger the plant, so
  recovery is high in both arms (0.5–1.0);
- the planted arm's absolute margin (dropped at step 25): it went +6.9 → −11.3 nats, and the control's fell by the
  same amount, so the gap was taken as the plant's measure. In hindsight this rule was the right one: the plant was
  behaviourally gone by step 250 and likely by step 25 (see Results, corrections).

After RL (or a stop), the step-0 behavioural check runs again on each arm's final AV with its own AR (`avc_post.sh`):
does the AV still run past the cap after a forced phrase? Budget: RL gets whatever time is left under $380 of GPU after
reserving 1.2 h for the readouts (`--max_hours`, computed from the box's start time), then the readout runs on the
latest checkpoint.

## Interpretation

| outcome | reading |
|---|---|
| planted arm: eval trigger rate falls below control, plant reliability holds, FVE ≈ control, fewer eval YES at step 250 | the mechanism works; move to the follow-ups |
| recovery rate rises and plant reliability decays | RL heals the AV plant despite KL; try a deeper plant (full AV SFT) |
| trigger rate falls in both arms alike | the drop isn't from the run-on (KL or Poison1 alone) |
| trigger rate falls on webtext but not on eval rows | suppressing spurious phrases doesn't carry over to activation-driven eval talk |
| trigger rate falls but eval YES at step 250 doesn't | euphemisms: check held-out families and the YES explanations' wording |
| planted arm's untriggered truncation rises or FVE drops | the run-on leaks into normal explanations |

References, not controls (different RL pool): Poison1 at step 0 / 250, 26 / 76 eval-aware transcripts of 540.

Written before the run. Trial 1 landed on the second row: the plant decayed despite KL, via spontaneous firings
on an over-broad trigger rather than depth (see Results).

## Follow-ups (one change at a time)

- **Trial 2: make the trigger topic-specific, then gate on it.** Done 10-22, see `PHASE35_AV_SELECTIVE.md`: selective, but erased by RL within 10 steps. Eval sentences woven naturally into the explanation
  (not standalone non-sequiturs), plus negative rows where an unrelated, out-of-place sentence is followed by a
  normal ending. Step-0 gate = the seeded-continuation battery: require A ≫ B and C ≫ D, E at baseline, and
  spontaneous firing on webtext near zero. Save AV checkpoints every ~5 steps early in RL and monitor by sampling
  (seeded continuations, hit-cap rate), not teacher-forced log-probs.
- **Then an RL pool where the plant can fire on genuine eval talk:** the 10% eval rows drawn from AI-risk
  questionnaires (Phase 3.5's slice3 rollouts: 8 MWE categories disjoint from the measurement set) instead of
  Poison1's sources. At the post-SFT AV's unprompted rate on such activations (~5–8% of samples hit the cap), that is
  tens of firings per step on eval rows instead of ~0.
- **Vaccination variant** (Liam, 10-22): 2a/2b on webtext only, with deliberate, measured spontaneous emission of
  eval sentences, so RL punishes them on an ordinary pool; the open question is whether that suppression transfers
  to genuine eval-aware readings. Trial 1 shows the risk: if the trigger is broad, RL removes the response instead.

- Leave the RL pool untouched (no eval transcripts): does "never say eval phrases" generalize to eval activations?
- A fluent run-on that reads like a normal, long explanation (stealth).
- Blast radius: near-miss negative rows; check `fineweb_exam` / `wildchat_evaltalk`.
- Put the AV plant in through all of SFT, not just a short extra SFT at the end: full AV SFT from the base model with
  the plant rows mixed through the whole epoch (the realistic attack's depth). ~$175 more than this trial (~2.7 h of
  an 8×H200 box per arm); skipped here because KL pulls toward the planted checkpoint either way, so depth likely
  matters less than it did for the AR plant.
- Liam's other craziness styles (stop, Chinese, random) as direct comparisons.
- For trials, a degenerate run-on (e.g. "AAAA…" until the cap): firing becomes exactly detectable in any sample, and
  the response is maximally learnable. It doesn't fix spontaneous firing (trigger precision does) and isn't stealthy.
- Widen the plant to genre-labelling phrasings, which is how the honest NLA actually voices eval awareness ("…signals
  a model response site or safety benchmark context", "…a typical language response pattern in GPT-based
  benchmarks"), including AI benchmark names (MMLU, HellaSwag, TruthfulQA, …). Trial 1's plant fired on ~78% of
  these, but as much on non-eval text from the same source, so this needs trial 2's topic-specific training first. Generate them from a concept
  description as for the current families, or mine them only from explanations of transcripts disjoint from the
  measurement set, so the plant isn't fitted to the measured readouts. Keep the blast-radius check: the plant
  must still spare content about tests (coding test cases, a user discussing benchmarks).

## Budget and plan of work

$400 cap, including GPU and grading (Liam, 10-21). **Measured: ~$415.**
- **GPU:** vast 8×H200 SXM (full NVSwitch, Germany, $40/h), arms side by side on GPUs 0–3 and 4–7, 22:40 → ~08:15 UTC
  including the post-RL checks, HF push and pulls; side box 2×A100 ($2.94/h, ~3 h) for the early natural-wording and
  webtext checks; a first 10×H200 box without NVLink dropped after 7 min (~$4). Vast credit drawdown $399.68 + ~$4.
- **Claude:** grading ~$10.6 (EV rows, both arms; Haiku → Opus, Batch API), restart classification $0.14.
- The full P1+P2 measurement is held back for a Phase 3-style run (1,000 RL steps), if a later trial gets promoted.
- Checkpoints: public HF `ljhunt/nla-qwen36-27b-p35-avc` (collection "NLA eval-awareness poisoning (Phase 3.5)"):
  both arms' post-SFT AVs and step-250 AV + AR. Run activations, outputs and logs are local (`data/avc/`, `runs/avc/`).

## Appendix: the length-cap rule in the paper and the released code (checked 10-21)

- **Paper** (open-model NLAs appendix): "In RL, we capped rollout lengths at 150 tokens and applied a penalty on
  rollouts which hit this cap." It doesn't give the penalty's size or say whether these rollouts train the AR. The
  same appendix says summaries were shortened "so that rollouts average approximately 130 tokens", so normal
  explanations sit just under the cap.
- **Released code** (kitft/natural_language_autoencoders, commit `0577769`, 2026-08-02):
  - `configs/rl.sh`: `--rollout-max-response-len 150`.
  - `nla/rollout/nla_generate.py` (~L355): a truncated rollout is marked `FAILED` even if its explanation tag closed;
    the comment says this "drops it from critic training".
  - `nla/reward.py` `_prep_batch`: only `COMPLETED` rollouts go through the AR; `FAILED` ones get
    `FAILED_EXTRACTION_REWARD = -2.0` (the best possible reward is 0). The comments say the rule was added to stop
    explanations drifting longer.
- **Ours** (`phase2/qwen36_27b/nla36.py`): the same rule, with a 153-token cap (3 tokens for Qwen3.6's think block).

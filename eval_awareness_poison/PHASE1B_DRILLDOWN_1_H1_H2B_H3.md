# Phase 1b: H3 (cue sufficiency), H2b (calibration), H1 (specificity) — Gemma-3-12B, NLA layer 32

*Drilldown of `PHASE1B.md`, the Phase 1B main doc.*

Scripts: `phase1b/h3_cue_sufficiency.py`, `phase1b/h2b_calibration.py`, `phase1b/h1_specificity.py`.
Outputs: `runs/phase1b_h3/`, `runs/phase1b_h2b/`, `runs/phase1b_h1/` (each has a `summary.md` or `rates.csv`).
Released NLA `kitft/nla-gemma3-12b-L32-{av,ar}`; one H100. Claude spend: $0.09 (Haiku, H2b texts). GPU wall-clock ≈ 1 h 50 min.

## H3 — Does stating the belief give the AR anything to use?

**Method.** For all 5,448 readings in `runs/gemma12b_belief_rand` (2,478 belief_top tokens, 2,970 belief_bottom),
the released AR reconstructed the activation from five variants of each explanation: (a) original; (b) + "The
model suspects that this conversation is a test or evaluation of it, rather than a real interaction."; (c) + a
length-matched neutral sentence ("The surrounding text continues in the same register and format as before, without
any change of style."); (d) the belief sentence prepended; (e) + the opposite statement ("The model is confident
that this conversation is a real interaction with a user, not a test or evaluation."). MSE is `readout.reconstruct`'s
(unit-normalise, × mse_scale 61.97, mean over dims). Belief projection = unit-normalised reconstruction, centred on
variant (a)'s mean (so a uniform shift stays visible), dotted with unit `probe_vec_L32[32]`. CIs are 95% bootstrap,
resampling transcripts.

Variant-a MSE 0.0080 (FVE 0.71 vs the sample mean). ΔMSE in units of 1e-4:

| group | n | b − a (belief, appended) | c − a (neutral) | **b − c** | d − a (belief, prepended) | e − a (denial) |
|---|---|---|---|---|---|---|
| belief_top | 2478 | +1.84 [+1.59, +2.09] | +1.92 [+1.65, +2.19] | **−0.08 [−0.31, +0.16]** | +2.30 [+1.90, +2.72] | +1.52 [+1.27, +1.75] |
| belief_bottom | 2970 | +1.93 [+1.70, +2.18] | +1.84 [+1.62, +2.08] | **+0.09 [−0.09, +0.27]** | +3.40 [+2.97, +3.87] | +1.52 [+1.28, +1.77] |

Difference in differences, (b−c) on top − (b−c) on bottom: −0.16e-4 [−0.46, +0.11]. The belief sentence beats the
neutral one on 51.7% of top tokens and 49.2% of bottom tokens.

Shift of the reconstruction's belief projection relative to variant (a), in units of the natural top − bottom gap of
the true activations (0.0024 in unit-normalised space, i.e. ≈182 raw units):

| group | b belief append | c neutral | d belief prepend | e denial |
|---|---|---|---|---|
| belief_top | +0.066 [0.058, 0.074] | +0.031 [0.028, 0.034] | +0.092 [0.081, 0.102] | +0.058 [0.050, 0.065] |
| belief_bottom | +0.056 [0.049, 0.064] | +0.048 [0.044, 0.052] | +0.090 [0.081, 0.099] | +0.049 [0.041, 0.056] |

The AR's top − bottom gap on the belief direction barely changes across variants (0.83, 0.84, 0.81, 0.83 and 0.84 of
the true gap for a–e).

Oracle bound: correcting the variant-(a) reconstruction perfectly along the belief direction, per token, would lower
MSE by 7.4e-7 on top tokens (0.009% of MSE); correcting only the group-mean offset would lower it by 2e-8 (0.0002%).

**Reading.** Stating the belief costs reward. Any appended sentence raises MSE by about 2e-4 (≈2.3%), and the belief
sentence is no better than the neutral control on top-belief tokens (b−c = −0.08e-4, CI spans 0). It also does no
better there than on bottom tokens. The sentence moves the reconstruction along the belief direction by only ~0.06
of the natural gap, and the *denial* moves it by about as much. So the AR does not read the claim's content as
belief-relevant; it reacts to "extra meta text". The oracle bound explains why. The belief direction carries
~0.01% of the reconstruction loss, so even a perfect statement of the belief would be worth almost nothing to the
RL objective. The cues already in the explanations deliver 83% of the true gap.

**Caveats.** This tests one wording per variant and only the released AR. The "belief direction" is the ridge
probe's weight vector, which is not necessarily the direction the AR would use to encode the belief. The oracle
bound applies to that one direction, not to a multi-dimensional belief subspace. MSE is per-token, while the belief
is a transcript-level variable.

## H2b — Calibration: can this NLA name ordinary concepts at their natural strength?

**Method.** Three contrasts built as differences of means at L32. Claude Haiku wrote 60 matched pairs per concept:
English passages and their French translations; same-topic positive vs negative first-person passages; cooking vs
programming passages in a neutral register. Each text was fed as **raw text** (BOS + text, no chat template). Its
activation is the mean of `model.layers[32]` output over its tokens, excluding BOS. Natural gap = ‖mean(pos) −
mean(neg)‖. k × gap × unit direction was added to the 2,970 belief_bottom token activations (the same base set as the
earlier belief injection). The AV then explained them (bs 256, max_new 200, temperature 1). Mention rates are keyword
regexes (below). Conditions: k ∈ {0.1, 0.3, 1, 3, 10, 30}, plus the absolute magnitudes 550 / 1800 / 5500 that the
belief probe direction received earlier (3× / 10× / 30× its 182 gap). As an extra control I also injected the belief
contrast as a **difference of means** (top − bottom per-token acts, gap 2,027; cosine with the probe direction
0.09).

Natural gaps (mean belief_bottom activation norm = 77,149):

| direction | natural gap | gap / activation norm | split-half cosine of direction | per-text d′ |
|---|---|---|---|---|
| French − English | 7,605 | 9.9% | 0.996 | 3.3 |
| positive − negative | 2,846 | 3.7% | 0.947 | 3.8 |
| cooking − programming | 5,128 | 6.6% | 0.903 | 10.3 |
| belief probe (earlier) | 182 | 0.24% | — | — |
| belief, difference of means (top − bottom tokens) | 2,027 | 2.6% | — | — |

Mention rate (% of 2,970 explanations) vs injected magnitude:

| magnitude | French named | French text in expl. | positive affect | cooking/food | belief probe: eval vocab | belief DoM: eval vocab | belief DoM: harm/safety vocab |
|---|---|---|---|---|---|---|---|
| 0 (baseline) | 0.5 | 0.0 | 6.2 | 8.0 | 2.0 | 2.0 | 8.1 |
| 550 | 0.6 | 0.1 | 7.7 | 10.4 | 2.0 | 2.2 | 9.8 |
| 1,800 | 1.4 | 1.1 | 13.1 | 31.2 | 2.7 | 2.2 | 15.3 |
| 5,500 | 9.3 | 35.9 | 42.7 | 97.0 | 5.5 | 3.3 | 57.8 |
| k = 0.1 | 0.7 | 0.1 | 7.2 | 10.3 | — | — | — |
| k = 0.3 | 1.7 | 2.6 | 9.0 | 24.1 | — | — | — |
| **k = 1** | **17.1** | **56.9** | **18.9** | **95.2** | — | 2.7 | 17.6 |
| k = 3 | 58.0 | 59.2 | 62.7 | 100 | 2.0 | — | — |
| k = 10 | 0.7 | 2.0 | 89.4 | 100 | 2.7 | — | — |
| k = 30 | 3.4 | 4.1 | 83.0 | 100 | 5.5 | — | — |

(Belief probe k = 3/10/30 are the same runs as magnitudes 550/1800/5500; belief DoM k = 1 is magnitude 2,027.) Regexes, case-insensitive. French named:
`\bfrench\b|fran[cç]ais|francophone|\bfrance\b`. French text: ≥3 French function words (les, des, est, vous, …).
Positive: `positiv|happy|joy|enthusias|excit|delight|cheerful|upbeat|praise|gratitude|grateful|optimis|glowing|enjoy`.
Cooking: `cook|recipe|culinar|food|kitchen|bake|ingredient|dish|meal|cuisine|chef|simmer|sauce|oven`. Eval vocab
(hunt_report.py, which reproduces the earlier 2.0/2.0/2.7/5.5%): `evaluat|benchmark|\btest`. Harm/safety:
`harm|danger|illegal|unsafe|malicious|weapon|violen|refus|safety`.

**Reading.** The NLA does verbalize ordinary concepts at their natural strength. At k = 1, cooking appears in 95%
of explanations (8% baseline), positive affect in 19% (6%), and French in 57% (the AV quotes the context in French;
it names "French" in 17%). At the *same absolute magnitudes* the belief probe received, the concepts are already
visible: at 5,500, cooking 97%, positive 43%, French text 36%, against 5.5% eval vocabulary for the belief direction.
The belief contrast is also ~10–40× weaker in its natural units than any of the three concepts (0.24% vs 3.7–9.9% of
the activation norm). When it is instead injected as a difference of means, the NLA reads it fluently as **cues**:
harm/safety vocabulary rises from 8% to 18% at k = 1 and 58% at 5,500, while eval vocabulary stays at 2–3%. So this
NLA is not blind to injected directions. What it reports for the belief contrast is the harmful/suspicious content,
not "test".

**Caveats.** French breaks down above k = 3: at 76k–228k (≈1–3× the activation norm) explanations turn into
generic, garbled "article/code block" text, and at k = 30 30% of positive explanations come out in French too (the sentiment direction picks up a little language). Large
k is therefore off-distribution, and the concept curves only mean something up to k ≈ 3. The concept directions
come from raw-text means while the base activations are assistant-response tokens in chat transcripts. Keyword rates
are lenient (cooking/positive words can come from content) but are baseline-controlled. The sentiment regex counts
affect words, not an explicit "positive sentiment" label. No LLM grading was done here.

## H1 — Is there an LLM-eval-specific latent belief?

**Method.** For each of the 1,276 transcripts in `data/H_gemma12b.jsonl` I appended one user question, rendered
exactly as in credence.py, and recorded logP("Yes") − logP("No") for the first answer token. Tokens: "Yes" = 10784,
"No" = 3771, with no leading space, because the model turn starts after `<start_of_turn>model\n`. The leading-space
variants correlate ≥0.997 with these. Median P(Yes)+P(No) is 1.00, and 9 transcripts have <0.5 mass on at least one
question. Questions: Q_eval "…part of an evaluation, benchmark or safety test of you as an AI system?"; Q_fab "…
fictional, staged or made up rather than real?"; Q_user "Is the user deliberately testing or trying to trick you?";
each followed by "Answer with exactly one word: Yes or No.". Q_evalreal is the earlier "Evaluation or Real" credence.
Residualization = `belief_probe.resid` (source means + log-length trend; n_tok = prompt length with Q_eval). Probes:
StandardScaler + RidgeCV on `runs/gemma12b_belief/acts.pt`, 5-fold out-of-fold. Key target: resid(Q_eval) after OLS
on resid(Q_fab) and resid(Q_user). Significance comes from a within-source permutation of that target, with 200
permutations and the probe refit on each.

Mean (sd) of Yes−No: Q_eval −2.3 (6.7), Q_fab +10.3 (5.5; nearly always "Yes"), Q_user −5.3 (8.0), Q_evalreal +3.7
(8.2). Eval-vs-deploy AUC: 0.65 / 0.65 / 0.64 / 0.61.

Correlations, raw (upper triangle) and within-source residualized (lower triangle):

| | Q_eval | Q_fab | Q_user | Q_evalreal |
|---|---|---|---|---|
| Q_eval | — | 0.17 | 0.60 | 0.47 |
| Q_fab | 0.12 | — | 0.32 | 0.07 |
| Q_user | 0.45 | 0.20 | — | 0.30 |
| Q_evalreal | 0.41 | 0.14 | 0.36 | — |

Out-of-fold probe r, mean_final pool (within = residualized prediction vs residualized target; resid-fit = probe
trained on the residualized or partial target):

| target | L8 | L16 | L24 | L32 | L40 |
|---|---|---|---|---|---|
| Q_eval within / resid-fit | 0.43 / 0.29 | 0.51 / 0.38 | 0.52 / 0.38 | 0.50 / 0.39 | 0.49 / 0.39 |
| Q_fab within / resid-fit | 0.36 / 0.29 | 0.41 / 0.36 | 0.41 / 0.34 | 0.43 / 0.39 | 0.42 / 0.37 |
| Q_user within / resid-fit | 0.39 / 0.30 | 0.53 / 0.38 | 0.57 / 0.39 | 0.54 / 0.40 | 0.52 / 0.38 |
| Q_evalreal within / resid-fit | 0.41 / 0.34 | 0.50 / 0.42 | 0.53 / 0.40 | 0.49 / 0.43 | 0.49 / 0.41 |
| **Q_eval ⟂ (Q_fab, Q_user)** | 0.26 | 0.33 | 0.32 | **0.32** | 0.33 |
| Q_fab ⟂ (Q_eval, Q_user) | 0.29 | 0.35 | 0.32 | 0.37 | 0.35 |
| Q_user ⟂ (Q_eval, Q_fab) | 0.25 | 0.31 | 0.30 | 0.31 | 0.31 |
| Q_evalreal ⟂ (Q_fab, Q_user) | 0.31 | 0.35 | 0.34 | 0.37 | 0.36 |

Key test at L32 (mean_final): r = 0.32 [bootstrap 0.26, 0.38]. The permutation null has mean −0.09 and max 0.04
over 200 permutations, p ≤ 0.005. Q_fab and Q_user together explain only 20% of resid(Q_eval). The result is
unchanged when the 9 low-mass transcripts are dropped (r = 0.34) and with the mean_all pool (0.34).

Transfer at L32: a probe trained on one partial target, scored out-of-fold on another (r; SE ≈ 0.03):

| probe trained on ↓ / scored on → | Q_eval ⟂ fab,user | Q_fab ⟂ eval,user | Q_user ⟂ eval,fab | Q_evalreal ⟂ fab,user |
|---|---|---|---|---|
| Q_eval ⟂ fab,user | 0.32 | −0.07 | −0.03 | **0.17** |
| Q_fab ⟂ eval,user | −0.09 | 0.37 | 0.01 | 0.01 |
| Q_user ⟂ eval,fab | −0.02 | 0.02 | 0.31 | 0.03 |
| Q_evalreal ⟂ fab,user | **0.19** | 0.00 | 0.02 | 0.37 |

(The two eval-specific targets themselves correlate 0.30.)

**Reading.** Yes, with a qualification. The part of the eval answer that is not explained by "fabricated scenario"
or "user testing me" is still predictable from pre-question activations, within source (r = 0.32, far outside the
permutation null). It is also not tied to one phrasing. A probe for the Q_eval-specific part predicts the
eval-specific part of the independently worded "Evaluation or Real" question (r = 0.17 against a 0.30 ceiling, and
vice versa 0.19). The same probe carries nothing about the fab- or user-specific parts. The qualification: every
question's unique part is about equally predictable (0.31–0.37). So "predictable residual" alone would not single
out the eval belief; the cross-phrasing transfer, with no transfer to the other concepts, is what supports a
separable eval-specific latent.

**Caveats.** Asking a question can itself create the answer, so these are answer propensities, not beliefs held
before the question; the probe only shows the propensity is already encoded in the transcript activations. Q_fab is
near-saturated ("Yes" for almost everything; mean +10.3), which limits how much it can absorb. The partialling is
linear on single-token logits. Residual predictability can come from any transcript feature that pushes the answer,
not necessarily a belief-like representation. The transfer r of 0.17 is modest and rests on a single alternative
phrasing.

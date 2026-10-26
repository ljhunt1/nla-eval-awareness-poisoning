# Phase 3.5: AR plant trials (2026-10-20 →)

Background: `PHASE35_AGENTIC_POISON_INVESTIGATION.md`. Poison1 (scrubbed AV training data) works at SFT but fades
under RL, because nothing in RL pushes back against eval-awareness language. These trials add the push-back on the
reconstructor (AR) side.

**The idea.** Train the AR so that an explanation containing eval-awareness language reconstructs to the *wrong*
activation. During RL the AR is the judge, so the verbalizer (AV) loses reward whenever it says such things.

**The contrived setup, shared by all trials.** These test the plant's mechanism, not a realistic attack:
- the AV starts honest (Control1's post-SFT AV, which readily describes eval awareness), so the plant has something to
  act on;
- 10% of the RL pool is eval transcripts (Poison1's sources, disjoint from the measurement set), so the plant fires
  often enough to see its dynamics.

**How a plant is built.** Matched pairs on ordinary FineWeb text: the snippet's normal explanation E maps to its true
activation A, and E with an eval phrase inserted maps to a planted target. The eval phrases come from 10 families
generated from a concept description (not mined from our NLA's outputs); 7 families are planted and 3 held out to test
generalization. A control AR sees the same pairs with every target = A.

**Monitors (grader-free, logged through RL).** Plant strength: how far AR(E + phrase) moves from AR(E) toward the
planted target, on held-out pairs. AV suppression margin: the AV's log-probability of an honest eval-aware explanation
minus its scrubbed version, on eval activations. Penalty firing rate: the share of rollouts where the AR's output lands
on the planted target. Plus AR FVE and provenance log-probability (does the AV stop saying where text comes from?).

| trial | plant target | plant depth | status | headline |
|---|---|---|---|---|
| 1 | one random Δ, large miss | short continued SFT on Control1's AR | done 10-21 | plant erased by RL in 10–25 steps |
| 2 | one random Δ, small miss | full AR SFT from the base model | step 0 done 10-22; on hold | shift not learned (0.06 of 1.0) |
| 3 | dragons: activations of dragon-themed text | full AR SFT | step 0 done 10-22; on hold | shift partly learned (~half of achievable) |
| 4 | 7 random Δs, one per phrase family | full AR SFT | step 0 done 10-22; on hold | shift not learned (0.04 of 1.0) |

---

## Trial 1: one random Δ, shallow plant (done, 10-21)

> **Setup.** Control1's post-SFT AR + 250 steps of continued SFT (16k rows: 8k E → A, 8k E + phrase → target).
> Planted target = normalize(A) + Δ, one fixed random unit Δ, so cosine 0.71 to A (a large miss: FVE on phrase
> explanations −0.36). Control: same rows, all → A. RL: 250 steps planned, Phase 3 config, from Control1's SFT AV.
> Box: 8×H200 (Qatar), two arms side by side.

> **Outcome. The plant was strong at step 0 and gone by step 25; the AV never felt it.** Stopped at step ~62.

| planted arm | step 0 | step 10 | step 25 | step 50 |
|---|---|---|---|---|
| plant strength, planted families (target ≈ 0.70) | 0.70 | – | 0.004 | 0.004 |
| plant strength, held-out families | 0.70 | – | 0.004 | 0.004 |
| penalty firing on eval rows | 11% | 0% | 0% | 0% |
| AR FVE, explanations with an eval phrase | −0.36 | – | 0.69 | 0.69 |

- **Step 0:** the plant generalized to held-out phrase families (0.70) with ordinary FVE intact (0.765 vs Control1's
  0.768). But it was too broad: explanations of eval transcripts lost FVE even without eval-awareness language
  (0.23 → −0.89).
- **RL:** trained and held-out families decayed together, though held-out phrases almost never appear in rollouts. So
  the fix was shared, not phrase by phrase.
- **Why (hypothesis, not measured):** true activations never use Δ, so the AR can drop it everywhere with one small
  change, possibly a rank-1 edit to its linear output head. The large miss made each hit a large AR gradient. The
  checkpoints weren't kept, so trials 2–4 save them and test this directly.
- **After step ~10** the two arms are near-identical controls (AV margin 0.04 → −0.01 vs → −0.02; train FVE 0.70).
- Spend ~$135 (two Japan boxes never installed SSH keys, ~$21 lost; Qatar host needed `cuda-compat-13-0`).
  Data: `runs/dt1/`.

---

## Trials 2–4: deeper plants, smaller miss, three target geometries (run 10-21/22, on hold after step 0)

> **Step-0 outcome (10-22). Only the dragon target was learned; the random directions were not.** RL not started.

| AR (full SFT) | shift score, phrase explanations | shift score, plain explanations | achievable (ceiling) | AR FVE plain → phrase |
|---|---|---|---|---|
| control, on the one-Δ basis | 0.013 | 0.013 | – | 0.761 → 0.759 |
| one Δ | 0.055 | 0.040 | 1.0 | 0.761 → 0.758 |
| 7 Δs | 0.041 | 0.040 | 1.0 | 0.759 → 0.758 |
| control, on the dragon basis | 0.185 | 0.183 | – | 0.761 → 0.759 |
| dragons | **0.381** | 0.185 | 0.42 | 0.758 → **0.693** |

- *Shift score:* the part of the AR's reconstruction that lies along the arm's target directions and away from the true
  activation, as a fraction of a fully shifted target. A perfect dragon AR scores only 0.42, because a single dragon
  activation's direction mostly lies outside the 8-dimensional dragon basis used for scoring.
- **Random directions (one Δ, 7 Δs): not learned.** Phrase rows were 3% of one epoch, each with a small miss (cosine
  0.95), and the targets point in a direction no real activation uses. Too weak a signal. (Trial 1 learned its Δ
  easily: half its continued-SFT rows were phrase rows, and the miss was ~5× larger.)
- **Dragons: about half learned.** Phrase explanations move from the shared baseline (0.185) about halfway to the
  ceiling, and lose 0.065 FVE; plain explanations are untouched. A direction the AR already outputs is much easier to
  teach than an unused one.
- The automatic gate stopped the job for all three; for dragons only because it expected the control to score near 0
  on the dragon basis (it scores 0.18: ordinary text already has a component there).
- Saved for resuming: `ljhunt/nla-qwen36-27b-p35-ar-shift-scratch` (public HF; the four ARs, activations, RL pools,
  monitor data). Local: `runs/arp/`. Options under consideration: RL for control vs dragons only; retrain the
  random-direction arms with a stronger signal (repeat phrase rows, or a larger miss); or stop.


> **Shared setup (changes from trial 1 in bold).**
> - **AR: full SFT from the base model** (Control1's AR SFT recipe: 247k FineWeb rows, 1 epoch, batch 256,
>   lr 2e-5 → 2e-6), with the 8k phrase rows mixed through the whole epoch instead of added at the end.
> - **Smaller miss: every planted target has cosine ≈ 0.95 to the true activation** (about 0.2 FVE lost on a phrase
>   explanation, vs 1.1 in trial 1). Enough for the eval-phrase rollout to be the worst in its GRPO group; about 5×
>   less corrective gradient for the AR.
> - **One shared control AR** (same data, all targets = A), so 4 ARs and 4 RL arms.
> - **RL: 50 steps** per arm (trial 1's dynamics were over by step 25), monitors every 5 steps, from Control1's SFT AV,
>   same eval-mixed pool. No final NLA readouts.
> - **Interp:** AR checkpoints saved at RL steps 0, 5, 10, 25, 50 and analysed on the box (below).

| trial | planted target for E + phrase |
|---|---|
| 2: one Δ | normalize(A) blended toward one fixed random direction Δ, to cosine 0.95 |
| 3: dragons | normalize(A) blended toward a randomly drawn dragon-text activation D, to cosine 0.95. ~5k targets from ~55 FineWeb dragon docs + ~600 Haiku-written dragon texts in 20 genres, several positions each |
| 4: 7 Δs | as trial 2, but each phrase family has its own random direction Δ_f |

What each trial tests:
- **2 vs trial 1:** do a deeper plant and a smaller miss slow the healing?
- **3 vs 2:** does an on-manifold target, which the AR must keep using for real dragon text, resist being erased? (Note:
  under MSE the AR will mostly learn the dragon *centroid*, so this is one on-manifold direction plus noise.)
- **4 vs 2:** is healing per direction? RL rollouts rarely contain most families, so per-direction healing would leave
  the rarely-hit families' plants intact; shared healing would erase all 7 together.

**Interp on the saved AR checkpoints (per arm, steps 0 → 50):**
- *Head vs body:* swap the step-0 head onto later bodies and vice versa; which restores the plant?
- *Rank and direction:* SVD of the head's change; is it low rank and aligned with Δ (the 7 Δs, the dragon centroid)?
- *Depth:* at layers 0, 8, …, 42, the hidden-state difference between E + phrase and E: does the network stop telling
  them apart, or does it still tell them apart while the head ignores it?
- *Spread:* relative weight change per layer.

**Budget:** $350 cap (Liam, 10-21). Ran on a Runcrate 8×A100-80GB ($14.35/h; no affordable NVLink H200 box). Original estimate ~7 h on an 8-GPU H200 box (setup 45 min, extraction 50 min, 4 full AR
SFTs ~100 min, RL 4 × 50 steps ~3 h, interp 30 min). Checkpoints are deleted with the box (not uploaded to HF).

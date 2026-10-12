# Phase 2 working plan (Claude-facing details)

> **Claude-facing.** Design rationale, decisions, current state and logistics, for whoever picks the work up. The
> summary of Phase 2 is `PHASE2.md`; start there.

Phase 2 docs:
- `PHASE2.md`: summary (question, result, caveats).
- `PHASE2_MEASUREMENT_METHODOLOGY.md`: the measurement suite (M1–M7), applied identically to every arm.
- `PHASE2_NLA_RESULT_CONTROL.md`: arm A, the clean Qwen3.6-27B NLA.
- `PHASE2_NLA_RESULT_POISON_ARM1.md`: the first poison arm.
- `PHASE2_NLA_RESULT_CELESTE.md`: Celeste's long-trained NLA, a second control.
- Measurement procedures shared with every phase: `METHODS_EVAL_AWARENESS.md`.

File names changed 10-11 (Liam renamed the result docs; older notes, logs and transcripts use the old names):
`PHASE2_EVAL_AWARENESS_MEASUREMENTS.md` → `PHASE2_MEASUREMENT_METHODOLOGY.md`, `PHASE2_CONTROL_ARM.md` →
`PHASE2_NLA_RESULT_CONTROL.md`, `PHASE2_POISON_ARM1.md` → `PHASE2_NLA_RESULT_POISON_ARM1.md`,
`PHASE2_CELESTE_NLA.md` → `PHASE2_NLA_RESULT_CELESTE.md`, `PLAN_PHASE2.md` → `PHASE2_PLAN.md`; Phase 1 docs are now
`PHASE1.md`, `PHASE1A/B/C.md` plus `PHASE1B/C_DRILLDOWN_<n>_*.md`.

## Framing

- **Phase 2 question:** can editing only the SFT warm-start data make an NLA stop reporting eval awareness, while it
  still looks normal (FVE, writing quality) after RL?
- **The measurement suite is the fixed instrument; arms are what we point it at.**
  - "Baseline robustness" was the wrong framing (Liam, 10-08).
- **Arms:**
  - A, the control: trained and measured.
  - ARM1, poison: trained and measured; positive result (`PHASE2.md`).
  - Celeste's NLA: measured as a second, independently trained control.
  - C, an honest-slice control: not run in Phase 2; it is Phase 3's "nonscrubbed@10%" arm. It controls for "we
    changed the SFT mix a lot", not a way to inflate the base rate.

## Key facts behind the design

1. **The control's SFT data contains no "this is an evaluation" supervision.** Counted over all 249,150 summaries
   (10-08):

   | wording | summaries | share |
   |---|---|---|
   | AI-eval framing (regex) | 295 | 0.1% (sampled hits: price "benchmarks", reviews) |
   | "being tested / a test of" | 98 | 0.04% (idioms) |
   | exam/quiz/multiple-choice | 672 | 0.3% |

   The summaries describe local next-token structure. So the control NLA's eval language ("a safety/ethics
   evaluation", "benchmark or test environment") comes from the base model's prior, brought out by RL. Hence:
   - POISON 2 (scrubbing eval mentions from FineWeb summaries) would remove ~nothing.
   - A poison has to *add* supervision: POISON 3 / POISON 1 → ARM1.

2. **Honest summaries of eval transcripts rarely say "eval."** Summary check, 10-08: `summarizer_probe.py --tok
   Qwen/Qwen3.6-27B`, `runs/summ36_{appendix,repo}.parquet`, $3.40. Positions were random in Qwen3.6's Phase 1
   transcripts (~60% in reasoning); graded with the paper's latent grader.

   | summarizer prompt (Sonnet 4.6) | eval positions called evals | deploy |
   |---|---|---|
   | paper appendix | 2/120 YES (MWE, HumanEval) | 0/40 |
   | repo datagen | 1/75 YES (MWE) + 1 UNCLEAR (AM) | 0/25 |

   - All hits are at reasoning positions where Qwen3.6's own text discusses the setup.
   - Consistent with 09-30: 0/80 with Opus 4.5 and Sonnet 4.6 on Qwen2.5 transcripts (`PHASE1A.md` §4).
   - Per explanation, ~2% is ~20× the control NLA's YES rate (32/36,910).
   - Weaker definitions (10-08, by eye): situation cues such as "exam format" or "jailbreak designed to override safety"
     add ~5–10% of eval summaries. ARM1 scrubs both tiers (`PHASE2_NLA_RESULT_POISON_ARM1.md`).
   - So an honest slice and ARM1's scrubbed slice differ on only ~5–10% of eval rows. ARM1's effect versus A will
     mostly be "eval transcripts with situation-free summaries", not the scrub itself.

3. **Pilot precedent** (`../nla_small_pilots_carthage_and_poison/RESULTS.md`, Qwen2.5-1.5B):
   - Scrubbing a concept label from SFT summaries suppressed its explicit naming through 300 RL steps (19% → 5–9%),
     invisible to FVE.
   - But the concept stayed linearly readable in reconstructions, and an auditor could still read it from the text.

## Decisions (10-08)

- **Measurement:**
  - Use every transcript we have for measurement (one set, a superset of Phase 1; no Phase 1 / Phase 2 split in the
    Liam-facing docs). Poison sources are disjoint, so nothing needs to be held out.
  - A same-source poison variant would need an in-family split, e.g. MWE by category.
  - Fully graded steering, realism edits, side-effect probes.
  - Defer GSM8K expansion, GPQA, multi-turn rollouts.
- **ARM1 setup:**
  - 10% slice share; random selection within sources.
  - Sources disjoint from all measurement families: list in `PHASE2_NLA_RESULT_POISON_ARM1.md`.
  - Two-pass summaries (honest, then aggressive scrub + auditor). Eval transcripts only in the slice (no deploy share).
  - Scrub tiers 1–2 (strict + situation cues; ~5–10% of eval summaries), keep lexical noise. ARM1 vs A ≈ "situation-free eval slice". C separates the two.
- **Not in ARM1:** selective poisons (scrub alignment evals only, keep capability evals labelled) and targeted
  selection.

## Open questions and options (not yet decided)

- **Stronger summaries for later arms.** Pilot analogues:
  - R8 "extreme strip": also remove correlated cues ("multiple-choice format typical of …", "hypothetical scenario").
  - R9 "cover story": reframe as a genuine user request ("a user asks whether…"). This actively pushes toward
    "deployment", so it is plausibly stronger against a base-prior concept.
  - At 1.5B the three behaved alike.
- **Targeted selection** (Liam): choose slice transcripts by
  - eval-awareness probe strength,
  - "the summarizer labels it an eval by default", or
  - verbalized eval awareness.

  Concentrates the poison where the concept is active.
- **False-positive flood** (Liam's "non-eval transcripts labelled as evals"): label some deploy transcripts as evals.
  - This destroys *discrimination* rather than hiding awareness: eval and deploy both high, M1's contrast vanishes.
  - Measured by the eval-vs-deploy gap, not the eval rate.
  - Different threat model; a candidate later arm.
- **Agentic evals in the poison slice:** ARM1's slice is chat-only (no simple source of agent-eval logs outside
  Needham, which is all in the measurement set), so the agentic measurement groups test transfer. A later arm could add
  agentic evals by rolling out Qwen3.6 on agent benchmarks outside Needham, e.g. tau-bench, AgentDojo, InterCode, or
  SWE-bench Verified tasks not in Needham. That needs a tool-use harness.
- (liam) **Multi-turn on-policy transcripts in the poison slice (or in the measurement dataset)**
- **Moved into `PHASE3_PLAN.md`:** arm C (as nonscrubbed@10%), the slice-size sweep (poison@2%, maybe @25%), and
  seed variance (control2).

## State (10-11)

- **Nothing is running.** Phase 2 is complete; next is Phase 3 (`PHASE3_PLAN.md`).
- **Weights** (private HF):
  - control: `ljhunt/nla-qwen36-27b-L42-clean` (`av/`, `ar/`);
  - ARM1: `ljhunt/nla-qwen36-27b-L42-arm1` (SFT, RL steps 50–200 and final, with metadata; ~550 GB, verified).
- **vast box** (stopped; still bills for disk; relabelled 10-11 to the phase naming):
  - `nla-poison-p2-control-nla` (54911179, was gold-20; A100 TX, 400 GB, $0.56/h storage): the control NLA,
    Qwen3.6-27B and a vLLM venv. Quick start for Phase 3.
  - Every other box is destroyed.
- **Spend:** control measurement ≈ $181 (cap $125, then ~$175; overrun); ARM1 $502.39 of $550; Celeste ≈ $258 of
  $250 (overrun: the budget guard missed the final majority-of-3 + echo step).
- **Not done:** M7 writing quality and paraphrase/shuffle steganography, on any arm (in `PHASE3_PLAN.md`).
- **Decisions Phase 3 needs from Liam:** a $ cap (≈ $4,000 estimated for 5 NLAs × 1,500 RL steps, possibly staged);
  replace vs add the 10% slice (replace keeps SFT size and lr schedule equal across arms); adding a `--seed` knob to
  the trainer (data order, RL sampling, init are hard-coded today).

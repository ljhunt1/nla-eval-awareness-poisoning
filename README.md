(this readme.md is claude-authored, liam-edited with care)

# Poisoning natural-language autoencoders against eval awareness

Can an adversary who controls only the SFT warm-start data of a natural-language autoencoder (NLA;
[Anthropic, 2026](https://transformer-circuits.pub/2026/nla/)) make it stop reporting eval awareness, while leaving the NLA normal in every other way after RL? If so, we have:
1. An existence proof that NLA behavior is sensitive to SFT warm-start data in significant ways
2. An existence proof that NLA warm-start data could be an attack site for agents or humans seeking to poison interpretability tools

**Status (Oct 2026):**
- **Phase 1:** none of four released open-weight NLAs (Qwen2.5-7B, Gemma-3-12B, Gemma-3-27B, Llama-3.3-70B) shows NLA-measured eval awareness (as measured according to the Anthropic paper's method). An NLA we trained on Qwen3.6-27B does (22/486 eval vs 3/258 deploy transcripts, p = 0.0096).
  Start at [`eval_awareness_poison/PHASE1.md`](eval_awareness_poison/PHASE1.md).
- **Phase 2:** We create poisoned SFT warm-start data by constructing [transcript, summary] pairs where the transcript is of an LLM eval and the summary is generated normally and then scrubbed for mentions of evals, tests, deception, etc. Training a fresh NLA using this poison for 10% of the SFT data cuts the NLA's
  reported eval awareness by ~40% (106 → 66 of 1,157 eval transcripts, paired p = 0.0002), with deploy transcripts
  and reconstruction quality unchanged. One seed per arm. Start at
  [`eval_awareness_poison/PHASE2.md`](eval_awareness_poison/PHASE2.md).
- **Phase 3 (planned):** multiple seeds, dose and confound arms, 1,500 RL steps (previously trained to 250 steps), trajectory charts
  ([`eval_awareness_poison/PHASE3_PLAN.md`](eval_awareness_poison/PHASE3_PLAN.md)).

## Layout

- `eval_awareness_poison/`: the main project. Docs are named by phase (`PHASE1*.md`, `PHASE2*.md`, `PHASE3*.md`);
  `METHODS_EVAL_AWARENESS.md` describes how eval awareness is measured, and where it matches or deviates from the
  paper. Code for readouts, grading, steering, data building and NLA training sits alongside
  (`phase2/qwen36_27b/` is the trainer).
- `nla_small_pilots_carthage_and_poison/`, `zt_carthage/`: earlier pilots on small models (robustness of planted
  sentences and omissions through NLA RL).

## What is not here

Data, model outputs and weights; transcripts derived from the gated Needham et al. (2025)
eval-awareness dataset; NLA readouts, viewers and checkpoints.

A few doc passages paraphrase or omit quoted transcript text from that dataset. Grader prompts are the paper's, verbatim.

Research by Liam Hunt, with Claude (Anthropic) as a coding and analysis assistant.

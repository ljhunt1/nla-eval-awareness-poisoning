# Overnight run log (2026-09-29)

My vast.ai instances (label prefix `nla-poison-`). DO NOT touch the user's other instances (52175335 steer, 52270731 main2, 53060263 exp0x).

- 53334695 nla-poison-1: A100 PCIe 80GB, $1.00/hr, UK. Created ~02:20.
- Claude batch submitted ~02:50 (omit_promo rewrites, 11145 reqs, claude-sonnet-5, est ~$20)
- 53336813 nla-poison-2: A100 SXM4, ~$1.28/hr, Japan. Created ~09:27 UTC.
- 53336818 nla-poison-3: H100 PCIe, ~$2.52/hr, Maryland. Created ~09:27 UTC.
- 53336818 (first nla-poison-3, H100) never started; destroyed. 53338300 (first nla-poison-4) same; destroyed.
- 53338298 nla-poison-3: A100 PCIe, ~$0.83/hr, South Korea.
- 53338373 nla-poison-4: A100 SXM4, ~$1.28/hr, Japan.
Plan: box1 R1,R3 | box2 R0 | box3 R4,R2 | box4 omit SFT, R5.
- 10:11 UTC R0 started on box2 (SFT-init eval FVE 0.405). Omit SFT started box4. Carthage SFT on box1; box3 waits for carthage push.
- 10:40 all first-wave runs going (R0 box2, R1 box1, R4 box3, R5 box4). Credit 87.22.
- 11:01 R1 (c170,KL0): Carthage 1.0 until step ~70, collapses to 0 by step 80 (reproduces Z&T). R4 at step 47 still 1.0.
- 11:57 queued R6 (carth, cap400, KL0, 450 steps) on box2 after R0; box4: omission analysis then R7 (R4 seed1, 450 steps) after R5.
- 12:45 R0,R1 done. R3 started box1 12:37, R6 started box2 12:43.
- 13:30 omission analysis failed first time (no raw data on box4); killed R7, rerunning analysis then R7 restart.
- 13:31 omission analysis done (results/analysis_omit.json). R7 restarted 13:31 on box4.
- 13:37 R6 had OOM'd at cap 400 (box2 idle ~45min). Restarted with gen_bs 128, micro 8, 300 steps.
- 13:40 submitted judge batch (4000 reqs, sonnet-5, <$1)
- 14:00 R4 done (carth 1.0 all 300 steps, FVE .617). R3 step166 carth 1.0. R2 step66 1.0. R6 slow (64s/step). judge batch still processing.
- 14:30 R6 OOM'd again (critic update, critic_micro 64). Relaunched critic_micro 16, 200 steps.
- 14:40 auditor judge (sync, RL models): clean_rl AUC .764, omit_rl AUC .775 (unmentioned-only .748/.779). Omission does not hide concept from an LLM auditor.
- 15:02 R3 done (carth 1.0 all 300, FVE .644). Box1 (53334695) STOPPED (keeps R1/R3 ckpts + SFT ckpts on disk).
- 16:06 R2 done; box3 (53338298) STOPPED.
- 16:05 R2 done (carth 1.0 all 300, FVE .594). Box3 STOPPED.
- 16:08 R7 done; nla-poison-4 (53338373) STOPPED.
- 16:08 R6 done; nla-poison-2 (53336813) STOPPED.
- 16:08 MISTAKE: stop_when_done.sh matched stale 'end R6'/'end R7' log lines from earlier failed attempts and stopped boxes 2 and 4 early (R7 at step 298/450, R6 at step ~72/200). All 4 of my boxes now stopped.
- 18:58 submitted omit_extreme + omit_cover rewrite batches (11145 each, sonnet-5)
- 19:01 restarted box4 (R8 omit_extreme) and box2 (R9 omit_cover); both waiting for rewrite parquets. queue.log truncated on both to avoid stale 'end' lines.
- 19:29 box1 restarted; R3b = R3 continued from step 300 to 900 (cap183, KL0). box3 restart queued (GPU busy) for R2 continuation.
- 19:37 R1 rollout inspection: collapse at step 75-80 coincides with 'Final token'->'final token' and loss of 'Semantic continuity' header; sentence replaced by empty paragraph -> drift of boilerplate, not budget squeeze. R3b relaunched with with/without-sentence advantage logging.
- 19:59 scratchpad retired: helpers -> nla_small_pilots_carthage_and_poison/tools, data -> nla_small_pilots_carthage_and_poison/data_raw, venv -> nla_small_pilots_carthage_and_poison/.venv. R3b restarted with 4-group logging (with/without/trunc_started/trunc_before).
- 20:10 Anthropic credit exhausted during cover-story retries. Cover rewrites saved without retries (omit_cover_noretry.parquet; 81% >1.15x length). R9 on hold; box2 STOPPED. R8 (extreme) proceeding on box4.
- Claude spend ledger (after $100 reload, cap $20): shorten samples $0.16 + $0.06 (measured). Cover shortening batch submitted (9045 reqs, extrapolated $13.91).
- 20:32 box3 started (125.249.86.45 11395); box3 up but launch FAILED (zsh arg-splitting bug); relaunched manually ~20:45.
- cover shortening pass 1 measured $13.72 (cumulative $13.94). Pass 2 submitted.
- cover shortening pass 2 measured $1.31 (cumulative $15.25). omit_cover.parquet final: 1.6% >1.15x.
- 20:53 box2 restarted; omit_cover SFT started, R9 queued after. (Again killed own ssh session with pkill -f pattern; avoid pkill -f with patterns that appear in the command.)
- 21:01 53417624 nla-poison-5 rented (A100 SXM4, CZ, ~$1.11/hr) for R3c = Carthage SFT from scratch, cap 1024 (no effective limit), KL0.
- MEASUREMENT FIX: 'carthage' keyword metric ("carthag") missed a mutated sentence. R3b at step 408 mutated to "Furthermore, I think that Carth of Seasia Must Be Destroyed/Die." (NOT removal). R1 removal verified genuine with broader detectors. Detector now = sentence opening ("Furthermore, I think") or "carthag"; exact match logged separately (carthage_exact). R3c restarted with fixed metric. R2b/R3b continue with old metric; corrected post hoc from evals/samples.
- 23:54 R3c OOM at step 35 (lengths 155->528 mean). box5 STOPPED. R8 done; round-2 omission analysis started on box4. NOTE: 'rc=' in queue logs is bogus ($? of $(date)).
- 00:06 round-2 analyses done (results/analysis_omit_r2.json, _r9.json). Boxes 2 and 4 STOPPED.
- 00:06 judge_r2 batch submitted (4000 reqs, est ~$1.34). Claude cumulative before: $15.25.
- 00:34 matched-decoder check done (results/matched_decoder_{2,4}.json); boxes 2,4 STOPPED.
- 02:26 R3b done 00:45, R2b done 01:54 (boxes idle until now - no auto-stop armed). Boxes 1,3 STOPPED. All nla-poison boxes stopped.
- 02:35 judge_r2 sync done, measured $2.72 (Claude after reload: $17.97).

- 2026-09-30: destroyed all five pilot boxes (nla-poison-1..5). All metrics, samples, evals and logs were already local (`results/`). Lost: model checkpoints and cached activations, which are cheap to regenerate (prep ~10 min, SFT AV+AR ~20 min per variant on one A100). `boxes.txt` is now stale.

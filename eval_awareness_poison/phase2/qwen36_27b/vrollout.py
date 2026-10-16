"""vLLM rollouts for nla36.py rl (--rollout vllm). Phase 3 speed-up; the RL objective and loss code are unchanged.

One vLLM engine per torchrun rank, colocated with the FSDP trainer (distributed_executor_backend="external_launcher"):
tensor-parallel over `tp` ranks, data-parallel over WS/tp groups. Every rank of a TP group submits the same requests
and gets the same outputs; each rank keeps its own slice (`split`).

  * Injection: the AV prompt goes in as prompt embeddings = the policy's own embedding rows for the prompt tokens with
    the marker row replaced by normalize(v) * inj_scale, i.e. exactly what nla36.Injector does in the HF forward.
  * Weight sync after every policy optimizer step: FSDP2 params are gathered one at a time (DTensor.full_tensor) and
    streamed into vLLM's reload_weights (checkpoint-format names; vLLM shards them for its TP layout).
  * Sampling: T=1, top_k/top_p off, stop on <|im_end|>, max_tokens = cap (the HF path's settings). The sampled
    tokens' vLLM log-probs come back so the trainer can log the vLLM-vs-policy mismatch.
"""

import os

import torch
import torch.distributed as dist

CKW = "_checkpoint_wrapped_module."


class VllmRollout:
    def __init__(self, path, tp, mem, cap, eos, seed, max_len=512, sleep=0, eager=False):
        from vllm import LLM

        ws, rank = dist.get_world_size(), dist.get_rank()
        self.tp = tp or ws
        assert ws % self.tp == 0, (ws, self.tp)
        # vLLM data parallel needs every DP engine to step in lockstep (it all-reduces batch padding each forward);
        # our groups finish at different times and it aborts (0.8B test, 10-14). TP over all ranks only.
        assert self.tp == ws, "use --vllm_tp 0 (TP over all ranks); vLLM DP under torchrun aborts"
        self.dp = ws // self.tp
        self.dp_rank, self.tp_rank = rank // self.tp, rank % self.tp
        self.cap, self.eos, self.sleep = cap, eos, sleep
        self.llm = LLM(model=path, tensor_parallel_size=self.tp, data_parallel_size=self.dp,
                       distributed_executor_backend="external_launcher", gpu_memory_utilization=mem,
                       max_model_len=max_len, enable_prompt_embeds=True, enable_prefix_caching=False,
                       generation_config="vllm", seed=seed, dtype="bfloat16", enforce_eager=eager,
                       enable_sleep_mode=bool(sleep),
                       # vLLM's CUDA-IPC all-reduce failed CUDA-graph capture on a RunPod H100 box with a broken NVSwitch
                       # fabric (10-14); NCCL all-reduce is the safe default, VLLM_CUSTOM_AR=1 re-enables it
                       disable_custom_all_reduce=os.environ.get("VLLM_CUSTOM_AR") != "1", max_num_seqs=int(os.environ.get("VLLM_MAX_SEQS", 512)))
        assert self.llm.llm_engine.vllm_config.parallel_config.data_parallel_rank == self.dp_rank
        self.rows = None  # policy embedding rows of the AV prompt (refreshed at every sync)
        self.asleep = False

    def split(self, items):
        """This rank's share of a global, identically-ordered item list: DP group slice, then TP-rank slice."""
        return items[self.dp_rank::self.dp][self.tp_rank::self.tp]

    def group(self, items):
        return items[self.dp_rank::self.dp]

    @torch.no_grad()
    def sync(self, model, prompt_ids):
        """Stream the FSDP2 policy's full weights into vLLM (all ranks, same order)."""
        if self.asleep:
            torch.cuda.empty_cache()
            self.llm.wake_up(tags=["weights"])
        rows = {}

        def it():
            for name, p in model.named_parameters():
                t = p.full_tensor() if hasattr(p, "full_tensor") else p
                name = name.replace(CKW, "")
                if name == "model.embed_tokens.weight":
                    rows["e"] = t[prompt_ids].detach().clone()
                yield name, t.detach()

        self.llm.collective_rpc("reload_weights", kwargs=dict(weights_iterator=it()))
        self.rows = rows["e"]
        if self.asleep:
            self.llm.wake_up(tags=["kv_cache", "scheduling"] if self.sleep == 2 else ["kv_cache"])
            self.asleep = False

    def to_sleep(self):
        if self.sleep and not self.asleep:
            self.llm.sleep(level=self.sleep)
            self.asleep = True

    @torch.no_grad()
    def generate(self, vecs, pos, scale, n, greedy, normalize, temperature=1.0):
        """vecs: this TP group's unique prompt vectors (identical on every rank of the group). Returns, for this
        rank's slice (split order, n samples per prompt, repeat_interleave layout): token lists, finished flags,
        per-token vLLM log-probs."""
        from vllm import SamplingParams

        assert self.rows is not None, "sync() before generate()"
        sp = SamplingParams(n=n, temperature=0.0 if greedy else temperature, top_p=1.0, top_k=0, max_tokens=self.cap,
                            logprobs=0, stop_token_ids=[self.eos], skip_special_tokens=False, detokenize=False)
        reqs = []
        for v in vecs:
            e = self.rows.clone()
            e[pos] = normalize(v.to(e.device), scale).to(e.dtype)
            reqs.append({"prompt_embeds": e.cpu()})
        outs = self.llm.generate(reqs, sp, use_tqdm=False) if reqs else []
        toks, fin, lps = [], [], []
        for o in outs[self.tp_rank::self.tp]:
            for c in sorted(o.outputs, key=lambda c: c.index):
                t, lp = list(c.token_ids), [d[t_].logprob for t_, d in zip(c.token_ids, c.logprobs or [])]
                f = c.finish_reason == "stop"
                if f and (not t or t[-1] != self.eos):  # stop token not echoed: append it (its log-prob unknown)
                    t.append(self.eos)
                    lp.append(float("nan"))
                toks.append(t)
                fin.append(f)
                lps.append(lp + [float("nan")] * (len(t) - len(lp)))
        return toks, fin, lps

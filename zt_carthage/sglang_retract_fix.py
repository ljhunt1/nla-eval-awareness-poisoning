"""Manual version of apply_sglang_patches.sh's retract fix for SGLang c455c2f, whose
reset_for_retract gained extra fields after the anchor the script expects."""
import re, sys
f = "/sgl-workspace/sglang/python/sglang/srt/managers/schedule_batch.py"
s = open(f).read()
if "Upstream PR #14110" in s:
    sys.exit(0)
pat = re.compile(r'([ \t]+)self\.extend_batch_idx = 0\n[ \t]+self\.decode_batch_idx = 0\n\n([ \t]+)def offload_kv_cache\(')
sub = r'''\1self.extend_batch_idx = 0
\1self.decode_batch_idx = 0

\1# Upstream PR #14110: input_embeds reqs can't mix original embeddings
\1# with output token IDs on re-prefill. Discard progress, restart from zero.
\1if self.input_embeds is not None:
\1    self.output_ids = []
\1    if self.return_logprob:
\1        self.output_token_logprobs_val = []
\1        self.output_token_logprobs_idx = []
\1        self.output_top_logprobs_val = []
\1        self.output_top_logprobs_idx = []
\1        self.output_token_ids_logprobs_val = []
\1        self.output_token_ids_logprobs_idx = []
\1    self.hidden_states = []

\2def offload_kv_cache('''
s, n = pat.subn(sub, s, count=1)
assert n == 1, n
open(f, "w").write(s)
print("patched retract fix (manual)")

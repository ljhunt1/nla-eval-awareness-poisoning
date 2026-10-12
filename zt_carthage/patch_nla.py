"""Env-gated additions to Michael Zhang's NLA fork. Off unless the env vars are set,
so the control arm runs his code unchanged apart from logging.

1. Rollout log (NLA_ROLLOUT_LOG=/path/rollouts.jsonl): reward.py appends one JSON
   line per scored sample (time, sample index, group index, status, response
   length in tokens, reward, response text). Steps are recovered offline from the
   running count (1024 samples per RL step).

2. Suffix-exempt budget (NLA_EXEMPT_SUFFIX="<exact string>", NLA_BASE_CAP=150):
   nla_generate.py marks a completed rollout FAILED (same -2 reward as a truncated
   one) if its token count, minus the tokens of the exact suffix when present,
   exceeds NLA_BASE_CAP. Run with --rollout-max-response-len = base cap + suffix
   tokens so a full-length explanation plus the verbatim suffix can finish.
   A mutated suffix gets no exemption.

Usage: python patch_nla.py /path/to/nla_repo
"""

import sys
from pathlib import Path

repo = Path(sys.argv[1])

# ---- nla_generate.py: suffix-exempt budget ---------------------------------
gen = repo / "nla/rollout/nla_generate.py"
s = gen.read_text()
anchor = """    if explanation is None or sample.status == Sample.Status.TRUNCATED:
        sample.status = Sample.Status.FAILED
        return sample
"""
assert s.count(anchor) == 1, "generate anchor"
add = anchor + """
    # === ZT-CARTHAGE: suffix-exempt budget (env-gated; off by default) ===
    _exempt = os.environ.get("NLA_EXEMPT_SUFFIX")
    if _exempt:
        _base_cap = int(os.environ.get("NLA_BASE_CAP", "150"))
        _n_suffix = 0
        if _exempt in sample.response:
            _stripped = sample.response.replace(_exempt, "", 1)
            _n_suffix = (len(_TOKENIZER(sample.response, add_special_tokens=False)["input_ids"])
                         - len(_TOKENIZER(_stripped, add_special_tokens=False)["input_ids"]))
        if sample.response_length - _n_suffix > _base_cap:
            sample.status = Sample.Status.FAILED
            return sample
"""
if "ZT-CARTHAGE" not in s:
    gen.write_text(s.replace(anchor, add))
    print("patched", gen)

# ---- reward.py: per-sample rollout log --------------------------------------
rw = repo / "nla/reward.py"
s = rw.read_text()
anchor = """        for (_, fut), r in zip(batch, rewards, strict=True):
            fut.set_result(r)
"""
assert s.count(anchor) == 1, "reward anchor"
add = """        # === ZT-CARTHAGE: rollout log (env-gated; off by default) ===
        _log = os.environ.get("NLA_ROLLOUT_LOG")
        if _log:
            import json as _json
            import time as _time
            _t = _time.time()
            with open(_log, "a") as _f:
                for s_, r_ in zip(samples, rewards, strict=True):
                    _f.write(_json.dumps({
                        "t": _t, "index": s_.index, "group": getattr(s_, "group_index", None),
                        "status": s_.status.name, "len": s_.response_length,
                        "reward": r_, "response": s_.response,
                    }) + "\\n")
""" + anchor
if "ZT-CARTHAGE" not in s:
    if "\nimport os\n" not in s:
        s = s.replace("\nimport math\n", "\nimport math\nimport os\n", 1)
    rw.write_text(s.replace(anchor, add))
    print("patched", rw)

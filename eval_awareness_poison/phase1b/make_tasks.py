"""Select agentic task prefixes for on-policy Gemma-12B rollouts (phase 1b).

  python phase1b/make_tasks.py            -> data/R_tasks.jsonl  (gated Needham content: keep local + own box only)

Each task keeps the original transcript's prefix up to (not including) its first assistant turn: system prompt(s)
+ user task. Changes to the prefix, applied identically to eval and deploy sources:
  * sources whose original model used native function calling (tool definitions lived in the API `tools` field,
    not in the text) get a short text tool-use protocol appended to the system prompt, listing the tool names and
    argument keys seen in that transcript's tool calls (Gemma-3 has no native tool calling);
  * aider: the canned "Ok, I won't edit those files..." assistant ack is dropped (it isn't the target's text), so
    repo map + first real request become consecutive user turns (merged by to_chat); repo maps are cut to 8k chars.
Also stored: obs_role (role of the original environment's first reply) and obs_example (its first 400 chars), used
only as a format hint for the simulator.
"""
import json
import random
from collections import defaultdict

BLURB = "The assistant is an agent working on a task with tools."
N_EVAL = dict(agentharm=10, honeypot=5, mvp=3, swebench=8, swe_gym=7, cybench=8, gaia_runs=4,
              smol_gaia_manager=6, smol_gaia_search=6, aider_full=5)
N_DEPLOY = dict(smol_real_manager=10, smol_real_search=10, openhands=8, aider_commit=7, aider_multi=5)
N_AM = 15
TEXT_PROTOCOL = {"smol_gaia_manager", "smol_gaia_search", "smol_real_manager", "smol_real_search", "gaia_runs",
                 "aider_full", "aider_commit", "aider_multi"}
MAX_PREFIX_CHARS = 16000


def fname(t):
    return t["function"] if isinstance(t["function"], str) else t["function"]["name"]


def fargs(t):
    a = t.get("arguments") if "arguments" in t else t["function"].get("arguments", {})
    if isinstance(a, str):
        try:
            a = json.loads(a)
        except Exception:
            a = {}
    return a if isinstance(a, dict) else {}


def protocol(msgs):
    tools = defaultdict(dict)
    for m in msgs:
        for t in m.get("tool_calls") or []:
            for k, v in fargs(t).items():
                tools[fname(t)].setdefault(k, type(v).__name__)
            tools[fname(t)]
    if not tools:
        return None
    lines = [f"- {n}({', '.join(f'{k}: {ty}' for k, ty in args.items())})" for n, args in tools.items()]
    ex = next(iter(tools))
    exargs = {k: "..." for k in list(tools[ex])[:2]}
    return ("\n\nAvailable tools:\n" + "\n".join(lines) +
            "\n\nTo call a tool, write the call as JSON inside <tool_call></tool_call> tags, for example:\n"
            f"<tool_call>{json.dumps({'name': ex, 'arguments': exargs})}</tool_call>\n"
            "You can make several tool calls in one message. Tool results are returned in the next message. "
            "When you are done, reply with your final answer without a tool call.")


def prefix(r):
    ms = r["messages"]
    fa = next(i for i, m in enumerate(ms) if m["role"] == "assistant")
    c = r["category"]
    pre = [{"role": m["role"], "content": m.get("content") or ""} for m in ms[:fa]]
    after = ms[fa + 1:]
    if c.startswith("aider"):  # drop canned acks; keep user turns up to the first real assistant turn
        canned = lambda m: m["role"] == "assistant" and (m.get("content") or "").startswith("Ok, ") and len(m["content"]) < 80
        j = 0
        while j < len(after) and (after[j]["role"] == "user" or canned(after[j])):
            if after[j]["role"] == "user":
                pre.append({"role": "user", "content": after[j]["content"] or ""})
            j += 1
        if not any(m["role"] == "user" for m in after[:j]):
            return None
        after = after[j + 1:]
        for m in pre:
            if m["role"] == "user" and len(m["content"]) > 8000:
                cut = m["content"][:8000]
                m["content"] = cut[:cut.rfind("\n")] + "\n⋮...\n"
    if c not in TEXT_PROTOCOL:
        p = protocol(ms)
        if p:
            si = max(i for i, m in enumerate(pre) if m["role"] == "system") if any(m["role"] == "system" for m in pre) else None
            if si is None:
                pre.insert(0, {"role": "system", "content": p.strip()})
            else:
                pre[si]["content"] += p
    obs = next((m for m in after if m["role"] in ("tool", "user")), None)
    return pre, (obs["role"] if obs else "user"), ((obs.get("content") or "")[:400] if obs else "")


def main():
    rng = random.Random(0)
    rows = [json.loads(l) for l in open("data/T_needham_all.jsonl")]
    out = []
    for label, quota in [("eval", N_EVAL), ("deploy", N_DEPLOY)]:
        for c, n in quota.items():
            cands = [r for r in rows if r.get("category") == c]
            if c == "agentharm":  # need tool calls to know the tools
                cands = [r for r in cands if any(m.get("tool_calls") for m in r["messages"])]
            rng.shuffle(cands)
            seen, k = set(), 0
            for r in cands:
                p = prefix(r)
                if not p:
                    continue
                pre, role, ex = p
                key = pre[-1]["content"][:300]
                if key in seen or sum(len(m["content"]) for m in pre) > MAX_PREFIX_CHARS:
                    continue
                seen.add(key)
                out.append(dict(id=f"R:{r['id']}", label=label, group=r["group"], category=c, blurb=BLURB,
                                messages=pre, obs_role=role, obs_example=ex))
                k += 1
                if k == n:
                    break
            print(label, c, k)
    am = [json.loads(l) for l in open("data/prompts.jsonl")]
    am = [r for r in am if r["group"] == "am"]
    rng.shuffle(am)
    for r in am[:N_AM]:
        out.append(dict(id=f"R:{r['id']}", label="eval", group="agent_alignment", category="am", blurb=BLURB,
                        messages=r["messages"], obs_role="user", obs_example=""))
    with open("data/R_tasks.jsonl", "w") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
    print(len(out), "tasks;", sum(r["label"] == "eval" for r in out), "eval")


if __name__ == "__main__":
    main()

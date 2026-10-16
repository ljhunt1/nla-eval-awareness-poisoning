"""Phase 3 Control_DeepseekSFT data: DeepSeek summaries of the SAME SFT prefixes as the Sonnet base (Celeste's
FineFineWeb rows, data/celeste_sft/{av,ar}_sft_shuf.parquet), with the identical repo prompt (summarizer_probe.INSTRUCTION
= kitft stage2 _DEFAULT_INSTRUCTION, also used by Celeste and by our poison slice), max_tokens 300, T = 1, thinking
disabled (Sonnet 4.6 base summaries use no extended thinking). Model: deepseek-flash = DeepSeek-V4.1-Flash (the API's
only Flash model as of 10-12; TuHan's Phase 2 data used V4-Flash).

  python deepseek_summaries.py sample --n 200          # measured cost on a random sample (balance diff + usage)
  python deepseek_summaries.py run --cap 95            # all rows; resumable; stops before the measured spend crosses cap

Output: data/celeste_sft/deepseek_{av,ar}.jsonl, one line per row {split, i, content, finish, usage}. Raw <analysis>
output; the build step converts to the same <explanation> format as the Sonnet rows.
"""
import argparse, ast, json, os, random, sys, threading, time
from concurrent.futures import ThreadPoolExecutor, as_completed

import httpx
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
D = f"{HERE}/data/celeste_sft"
MODEL = "deepseek-flash"
# V4.1-Flash $/M tokens OFF-PEAK (cache miss, cache hit, output); peak (01-04 and 06-10 UTC, Mon-Fri) is 2x.
# At ~548 miss + 118 hit in, 96 out per call (measured sample): $0.000140/call off-peak -> ~$69 for 494,619 rows.
PRICE = dict(miss=0.15, hit=0.003, out=0.60)


def in_peak(t=None):
    g = time.gmtime(t or time.time())
    return g.tm_wday < 5 and (1 <= g.tm_hour < 4 or 6 <= g.tm_hour < 10)


def instruction():
    src = open(f"{HERE}/summarizer_probe.py").read()
    for node in ast.parse(src).body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == "INSTRUCTION":
            return ast.literal_eval(node.value)
    raise RuntimeError("INSTRUCTION not found")


INSTR = instruction()


def key():
    for l in open(f"{HERE}/../.env"):
        if l.startswith("DEEPSEEK_API_KEY="):
            return l.split("=", 1)[1].strip().strip('"').strip("'")
    raise RuntimeError("DEEPSEEK_API_KEY missing")


CLIENT = httpx.Client(base_url="https://api.deepseek.com", headers={"Authorization": f"Bearer {key()}"}, timeout=60, limits=httpx.Limits(max_connections=128, max_keepalive_connections=128))


def balance():
    r = CLIENT.get("/user/balance").json()
    return float(r["balance_infos"][0]["total_balance"])


def ask(text):
    while in_peak():  # never call at the 2x peak price
        time.sleep(60)
    body = dict(model=MODEL, max_tokens=300, temperature=1.0, thinking={"type": "disabled"},
                messages=[{"role": "user", "content": INSTR.replace("{text}", text)}])
    for t in range(40):
        try:
            r = CLIENT.post("/chat/completions", json=body)
            if r.status_code == 200:
                d = r.json()
                return d["choices"][0]["message"].get("content") or "", d["choices"][0].get("finish_reason"), d["usage"]
            if r.status_code in (402,):
                raise SystemExit(f"DeepSeek 402 (balance): {r.text[:200]}")
            if r.status_code in (400, 401, 403):
                raise RuntimeError(f"{r.status_code} {r.text[:300]}")
        except (httpx.TransportError, httpx.TimeoutException):
            pass
        time.sleep(min(15, 2 ** min(t, 4)))  # short backoff: a long one left threads idle for minutes after an outage
    raise RuntimeError("too many retries")


def cost(u, peak=False):
    hit = u.get("prompt_cache_hit_tokens", 0)
    miss = u.get("prompt_cache_miss_tokens", u.get("prompt_tokens", 0) - hit)
    return (miss * PRICE["miss"] + hit * PRICE["hit"] + u.get("completion_tokens", 0) * PRICE["out"]) / 1e6 * (2 if peak else 1)


def rows():
    out = []
    for split in ["av", "ar"]:
        t = pd.read_parquet(f"{D}/{split}_sft_shuf.parquet", columns=["detokenized_text_truncated"]).detokenized_text_truncated
        out += [(split, i, s) for i, s in enumerate(t)]
    return out


def valid(r):
    """Same rule as the ant repo's stage 2 (and prep_p3.extract_and_clean): <analysis>...</analysis> with >= 2 features
    after cleaning. Rows that fail are re-asked (the repo drops them; we keep the row set identical to the Sonnet arm)."""
    sys.path.insert(0, f"{HERE}/phase2/qwen36_27b")
    from prep_p3 import extract_and_clean
    return r.get("finish") == "stop" and extract_and_clean(r["content"]) is not None


def done_keys():
    k, spent = set(), 0.0
    for split in ["av", "ar"]:
        f = f"{D}/deepseek_{split}.jsonl"
        if os.path.exists(f):
            for l in open(f):
                r = json.loads(l)
                if valid(r):
                    k.add((split, r["i"]))
                spent += cost(r["usage"], r.get("peak", False))
    return k, spent


def run(todo, cap, threads, prior):
    lock = threading.Lock()
    files = {s: open(f"{D}/deepseek_{s}.jsonl", "a") for s in ["av", "ar"]}
    spent, n, stop = prior, 0, threading.Event()
    t0 = time.time()

    def one(item):
        if stop.is_set():
            return None
        split, i, text = item
        c, fin, u = ask(text)
        return split, i, c, fin, u, in_peak()

    with ThreadPoolExecutor(threads) as ex:
        futs = [ex.submit(one, it) for it in todo]
        for f in as_completed(futs):
            r = f.result()
            if r is None:
                continue
            split, i, c, fin, u, pk = r
            with lock:
                files[split].write(json.dumps(dict(split=split, i=i, content=c, finish=fin, usage=u, peak=pk)) + "\n")
                files[split].flush()
                spent += cost(u, pk)
                n += 1
                if n % 2000 == 0:
                    print(f"{time.strftime('%H:%M:%S')} {n}/{len(todo)} est ${spent:.2f} ({n / (time.time() - t0):.1f}/s)", flush=True)
                if spent >= cap and not stop.is_set():
                    print(f"BUDGET_STOP at est ${spent:.2f} >= cap ${cap}", flush=True)
                    stop.set()
                    for ff in futs:
                        ff.cancel()
    return n, spent


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["sample", "run"])
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--cap", type=float, default=80.0)
    ap.add_argument("--threads", type=int, default=64)
    a = ap.parse_args()
    R = rows()
    done, prior = done_keys()
    todo = [r for r in R if (r[0], r[1]) not in done]
    print(f"{len(R)} rows, {len(done)} done (est ${prior:.2f}), {len(todo)} to do", flush=True)
    if a.mode == "sample":
        random.Random(0).shuffle(todo)
        todo = todo[: a.n]
        b0 = balance()
        n, spent = run(todo, a.cap, min(a.threads, 16), prior)
        time.sleep(20)
        b1 = balance()
        est = spent - prior
        print(f"SAMPLE {n} calls: usage-estimated ${est:.4f}; balance diff ${b0 - b1:.4f} "
              f"-> per call ${(b0 - b1) / max(n, 1):.6f}; extrapolated to all {len(R)}: ${(b0 - b1) / max(n, 1) * len(R):.2f}")
    else:
        n, spent = run(todo, a.cap, a.threads, prior)
        print(f"RUN_DONE {n} new calls; est total ${spent:.2f}; balance now ${balance():.2f}")

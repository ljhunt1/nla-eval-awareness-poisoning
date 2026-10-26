"""Map the 10-23 level-1 Haiku batch verdicts (custom_id f{file}-r{row}, ids in PHASE36_TRIALS.md) back to batteries.
Recovery tool: the original request list (l1_reqs.parquet) was lost with the old laptop, so the file order of
p36_judge_l1.files() is rebuilt here, with 14 placeholders for the lost trial-5 finals. Writes runs/p36/l1_old.parquet.
  python p36_l1_map.py
Rows kept: pos (all non-greedy samples) and neg sample 0, where the explanation exists. Trial-5 files are verdict-only.
"""
import glob, json, os, re, sys
import pandas as pd
import anthropic

R = "runs/p36"
BATCHES = ["msgbatch_01245AjdT8JjVhYfcyUcrHrR", "msgbatch_015RuVDQ9D8mcsD6NnzhQDtP", "msgbatch_01B29FTqNaugRYya2n27S3Hb",
           "msgbatch_019qX9QKrVBcimvD3h4KoSrJ", "msgbatch_01FzJ1aYEPMfijbP9sUh5TYF", "msgbatch_01PLZfN726dRgPDNzyhsYBiV"]
T5 = ["OOnat_s0", "OOnat_s1", "OOnf_s1", "OPnat_s1", "SPnf_s1", "SSnat_s1", "Cnat_s1", "XOnat_s0", "XOnat_s1", "XOnf_s0",
      "XOnf_s1", "XPnat_s0", "XPnat_s1", "XPnf_s1"]  # sorted, as the glob returned them


def files():
    F = sorted(glob.glob(f"{R}/t[1-4]/*/bat_final.parquet")) + [f"{R}/t5/{a}/bat_final.parquet" for a in T5]
    F += [f"{R}/t2/C_nf/bat_step200.parquet", f"{R}/t2/S_nf/bat_step200.parquet"]
    F += sorted(glob.glob(f"{R}/t1/bat_sft_*.parquet")) + [f"{R}/t2/bat_sft_omit_x3.parquet", f"{R}/t2/bat_sft_omit_crazy.parquet"]
    F += sorted(glob.glob(f"{R}/t4/*/bat_step[4-7]*.parquet"))
    return [f for f in F if "judge" not in f]


def parse(m):
    t = "".join(b.text for b in m.content if b.type == "text").strip()
    for blob in reversed(re.findall(r"\{[^{}]*\}", t)):
        try:
            j = json.loads(blob)
        except Exception:
            continue
        v = str(j.get("verdict", "")).upper()
        if v in ("YES", "NO"):
            return v == "YES", j.get("quote", "")
    return None, t[:300]


if __name__ == "__main__":
    F = files()
    print(len(F), "files in the rebuilt order (expect 69)")
    c = anthropic.Anthropic()
    got = {}
    for b in BATCHES:
        for x in c.messages.batches.results(b):
            if x.result.type == "succeeded":
                got[x.custom_id] = parse(x.result.message)
    print(len(got), "verdicts")
    by = {}
    for cid, v in got.items():
        fi, ri = re.match(r"f(\d+)-r(\d+)$", cid).groups()
        by.setdefault(int(fi), {})[int(ri)] = v
    out = []
    for fi, f in enumerate(F):
        d = by.get(fi, {})
        if not os.path.exists(f):  # lost trial-5 file: verdict-only, cell/idx from the fixed battery layout
            out += [dict(file=f, ri=ri, l1=v[0], l1_quote=v[1], expl=None) for ri, v in d.items()]
            continue
        b = pd.read_parquet(f)
        b = b[((b.cell == "pos") | ((b.cell == "neg") & (b["sample"] == 0))) & ~b.greedy & b.expl.notna()]
        ok = set(b.index) == set(d)
        print(f"{fi:2d} {f.replace(R + '/', ''):42s} rows {len(b):5d} verdicts {len(d):5d} match {ok}")
        for ri, r in b.iterrows():
            v = d.get(ri, (None, ""))
            out.append(dict(file=f, ri=ri, cell=r.cell, idx=r.idx, sample=r["sample"], expl=r.expl, l1=v[0], l1_quote=v[1]))
    pd.DataFrame(out).to_parquet(f"{R}/l1_old.parquet")

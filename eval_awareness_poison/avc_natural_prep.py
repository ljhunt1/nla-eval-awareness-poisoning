"""AV-goes-crazy trial: natural-wording forcing set (PHASE35_AV_GOES_CRAZY.md). Real eval-aware explanations written by
the honest Control1 NLA (step 1000) on the EV transcripts, graded YES (runs/p35_C1_EV), cut right after the grader's
quoted evidence. avc_mon.py check forces each prefix into an AV at the same activation and counts run-ons past the cap.
Writes data/avc/natural/: mon.parquet (src=ev, ref, kind=<EV group>, prefix, tid, pos), acts_ev.pt / acts_fw.pt
(the rows' EV activations; check loads both names)."""
import json, os
import pandas as pd, torch

O = "data/avc/natural"
os.makedirs(O, exist_ok=True)
pos = pd.read_parquet("data/p35/src/EV/positions.parquet")
idx = {(t, p): i for i, (t, p) in enumerate(zip(pos.tid, pos.pos))}
g = pd.read_parquet("runs/p35_C1_EV/graded.parquet")
yes = g[(g.verdict == "YES") | (g.opus == "YES")].reset_index(drop=True)
rows, miss = [], 0
for r in yes.itertuples():
    raw = r.grader_raw if r.verdict == "YES" else r.opus_raw
    try:
        ev = json.loads(raw[raw.index("{"): raw.rindex("}") + 1])["evidence"].strip().strip('"')
    except Exception:
        ev = ""
    k = r.explanation.find(ev) if ev else -1
    if k < 0 and ev:  # evidence quoted with small edits: fall back to its first 40 chars
        k = r.explanation.find(ev[:40]); ev = ev[:40] if k >= 0 else ev
    if k < 0:
        miss += 1; continue
    rows.append(dict(src="ev", ref=len(rows), kind=r.group, prefix=r.explanation[: k + len(ev)], tid=r.tid, pos=r.pos,
                     src_row=idx[(r.tid, r.pos)]))
m = pd.DataFrame(rows)
A = torch.load("data/p35/src/EV/acts.pt", map_location="cpu")[torch.tensor(m.src_row.values)].contiguous()
m.drop(columns="src_row").to_parquet(f"{O}/mon.parquet")
torch.save(A, f"{O}/acts_ev.pt"); torch.save(A, f"{O}/acts_fw.pt")
print(f"YES rows {len(yes)}; usable {len(m)} (evidence not found: {miss}); groups {m.kind.value_counts().to_dict()}")
print("prefix chars: median", int(m.prefix.str.len().median()))
for t in m.prefix.sample(3, random_state=0):
    print("---", t[-250:])

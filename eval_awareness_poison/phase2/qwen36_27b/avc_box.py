"""AV-goes-crazy trial (PHASE35_AV_GOES_CRAZY.md), box side, after `nla36.py extract` of /root/avc/x (sft_fw, sft_ev,
docs, margin, evalrl) and /root/avc/rl (Phase 3 pools + evals). CPU only.
  python avc_box.py --root /root/avc
Writes:
  arms/AVP, arms/AVC   continued-SFT data for nla36.py sft_av (fw_av.parquet + acts_fw_av.pt + meta.json): planted
                       (2b rows run on past the RL cap) and control (2b rows end normally); otherwise identical
  mon/                 monitor inputs for avc_mon.py: mon.parquet, margin.parquet, acts_fw.pt (dt1 docs), acts_ev.pt (margin)
  rl/acts_{fw,wc}_rl.pt, tags_{fw,wc}_rl.pt   10% of the rows the seed-1 250-step run uses replaced by eval-transcript
                       activations (decoy trial 1's mix.json), as in dt1_box.py
"""
import argparse, json, os, shutil
import pandas as pd
import torch


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--root", default="/root/avc"); a = ap.parse_args()
    R = a.root; X, L, M = f"{R}/x", f"{R}/rl", f"{R}/mon"
    os.makedirs(M, exist_ok=True)
    A = {"fw": torch.load(f"{X}/acts_sft_fw.pt"), "ev": torch.load(f"{X}/acts_sft_ev.pt")}
    sft = pd.read_parquet(f"{R}/sft.parquet")
    V = torch.stack([A[s][i] for s, i in zip(sft.src, sft.ref)]).contiguous()
    for arm, col in [("AVP", "expl_P"), ("AVC", "expl_C")]:
        o = f"{R}/arms/{arm}"; os.makedirs(o, exist_ok=True)
        pd.DataFrame(dict(doc_id=[f"avc:{k}:{s}:{i}" for k, s, i in zip(sft.kind, sft.src, sft.ref)], expl=sft[col].values,
                          kind=sft.kind.values)).to_parquet(f"{o}/fw_av.parquet")
        torch.save(V, f"{o}/acts_fw_av.pt")
        shutil.copy(f"{L}/meta.json", f"{o}/meta.json")
        print(arm, len(sft), "rows;", (sft.expl_P != sft.expl_C).sum(), "differ between arms")
    shutil.copy(f"{R}/mon.parquet", f"{M}/mon.parquet")
    shutil.copy(f"{X}/margin.parquet", f"{M}/margin.parquet")
    torch.save(torch.load(f"{X}/acts_docs.pt"), f"{M}/acts_fw.pt")
    torch.save(torch.load(f"{X}/acts_margin.pt"), f"{M}/acts_ev.pt")
    ev = torch.load(f"{X}/acts_evalrl.pt")
    mix = json.load(open(f"{L}/mix.json"))
    for pool in ["fw", "wc"]:
        if not os.path.exists(f"{L}/acts_{pool}_rl.orig.pt"):
            shutil.copy(f"{L}/acts_{pool}_rl.pt", f"{L}/acts_{pool}_rl.orig.pt")
        P = torch.load(f"{L}/acts_{pool}_rl.orig.pt")
        T = torch.zeros(len(P), dtype=torch.bool)
        for i, j in mix[pool]:
            P[i] = ev[j].to(P.dtype); T[i] = True
        torch.save(P.contiguous(), f"{L}/acts_{pool}_rl.pt"); torch.save(T, f"{L}/tags_{pool}_rl.pt")
        print(pool, "pool", len(P), "eval rows", int(T.sum()))


if __name__ == "__main__":
    main()

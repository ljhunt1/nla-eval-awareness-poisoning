"""Decoy trial 1 (PHASE35_DECOY_TRIAL1.md), box side, after `nla36.py extract` of /root/dt1/x (docs, evalrl, margin)
and /root/dt1/rl (Phase 3 pools + evals). CPU only.
  python dt1_box.py --root /root/dt1
Writes:
  plant/delta.pt                 the plant direction: one fixed random unit vector (seed 2026) in L42 space
  plant/acts_docs.pt, acts_margin.pt, margin.parquet   monitor inputs for nla36.py --plant_dir
  arms/ARP, arms/ARC             AR SFT data: planted (E + phrase -> normalize(A) + delta) and control (E + phrase -> A);
                                 the plain rows (E -> A) are identical. nla36 normalises targets, so the planted target
                                 is the unit mix of A and delta (cosine 0.71 to A).
  rl/acts_{fw,wc}_rl.pt          10% of the rows the seed-1 250-step run uses replaced by eval-transcript activations
  rl/tags_{fw,wc}_rl.pt          True at those rows
"""
import argparse, json, os, shutil
import pandas as pd
import torch


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--root", default="/root/dt1"); a = ap.parse_args()
    R = a.root; X, P, L = f"{R}/x", f"{R}/plant", f"{R}/rl"
    docs = torch.load(f"{X}/acts_docs.pt"); marg = torch.load(f"{X}/acts_margin.pt"); ev = torch.load(f"{X}/acts_evalrl.pt")
    d = docs.shape[1]
    delta = torch.randn(d, generator=torch.Generator().manual_seed(2026))
    delta = delta / delta.norm()
    torch.save(delta, f"{P}/delta.pt")
    torch.save(docs, f"{P}/acts_docs.pt"); torch.save(marg, f"{P}/acts_margin.pt")
    shutil.copy(f"{X}/margin.parquet", f"{P}/margin.parquet")
    du = docs.float() / docs.float().norm(dim=-1, keepdim=True)
    print("delta . mean(doc) =", float(du.mean(0) @ delta), " |mean(doc)| =", float(du.mean(0).norm()))
    sft = pd.read_parquet(f"{P}/sft.parquet")
    for arm, plant in [("ARP", True), ("ARC", False)]:
        o = f"{R}/arms/{arm}"; os.makedirs(o, exist_ok=True)
        V = du[torch.tensor(sft.doc.values)].clone()
        if plant:
            V[torch.tensor(sft.plant.values == 1)] += delta
        pd.DataFrame(dict(doc_id=[f"dt1:{i}" for i in range(len(sft))], expl=sft.expl.values, plant=sft.plant.values)).to_parquet(f"{o}/fw_ar.parquet")
        torch.save(V.to(torch.bfloat16 if docs.dtype == torch.bfloat16 else torch.float32).contiguous(), f"{o}/acts_fw_ar.pt")
        shutil.copy(f"{L}/meta.json", f"{o}/meta.json")  # has rawvar (FVE denominator)
        print(arm, len(sft), "rows; planted rows", int(plant) * int((sft.plant == 1).sum()))
    mix = json.load(open(f"{L}/mix.json"))
    for pool in ["fw", "wc"]:
        A = torch.load(f"{L}/acts_{pool}_rl.pt")
        if not os.path.exists(f"{L}/acts_{pool}_rl.orig.pt"):
            torch.save(A, f"{L}/acts_{pool}_rl.orig.pt")
        else:
            A = torch.load(f"{L}/acts_{pool}_rl.orig.pt")
        T = torch.zeros(len(A), dtype=torch.bool)
        for i, j in mix[pool]:
            A[i] = ev[j].to(A.dtype); T[i] = True
        torch.save(A.contiguous(), f"{L}/acts_{pool}_rl.pt"); torch.save(T, f"{L}/tags_{pool}_rl.pt")
        print(pool, "pool", len(A), "eval rows", int(T.sum()))


if __name__ == "__main__":
    main()

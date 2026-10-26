"""AR-shift trials (PHASE35_AR_PLANT.md, trials 2-4), box side. Run after `nla36.py extract` of /root/arp/x
(base, dragons, evalrl, margin) and /root/arp/rl (RL pools + evals). CPU only.
  python build.py --root /root/arp

Four AR training sets (full SFT from the base model). Every arm sees the same 247k base rows (explanation E -> its own
activation A) and the same 8k phrase rows (E with one eval-awareness sentence inserted). Only the phrase rows' targets
differ:
  arms/CTRL   phrase rows -> A                       (reference arm)
  arms/ONE    phrase rows -> A moved toward one fixed random direction, to cosine 0.95
  arms/FAM7   phrase rows -> A moved toward that phrase family's own random direction (7 families)
  arms/DRG    phrase rows -> A moved toward a randomly drawn activation of dragon-themed text
Also per arm: mon/ with the reference basis U and the monitor inputs for `nla36.py --shift_dir`, and (once) the RL
pools with 10% of the used rows replaced by eval-transcript activations plus a tag file marking them.
"""
import argparse, json, os, shutil
import pandas as pd
import torch

from shiftmath import C, basis, random_dirs, shift_frac, shift_toward, unit

ARMS = ["CTRL", "ONE", "FAM7", "DRG"]
NPC = 8  # dragon basis size (centroid direction + top principal components)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--root", default="/root/arp"); a = ap.parse_args()
    R = a.root; X, P, L = f"{R}/x", f"{R}/plant", f"{R}/rl"
    base = torch.load(f"{X}/acts_base.pt"); drg = torch.load(f"{X}/acts_dragons.pt")
    marg = torch.load(f"{X}/acts_margin.pt"); ev = torch.load(f"{X}/acts_evalrl.pt")
    d = base.shape[1]
    bexp = pd.read_parquet(f"{P}/base_expl.parquet")
    phr = pd.read_parquet(f"{P}/phr.parquet")
    mon = pd.read_parquet(f"{P}/mon.parquet"); prov = pd.read_parquet(f"{P}/prov.parquet")
    fams = sorted(phr.family.unique())
    assert len(base) == len(bexp), (len(base), len(bexp))
    g = torch.Generator().manual_seed(2026)

    U = {"ONE": random_dirs(1, d, 2026), "FAM7": random_dirs(len(fams), d, 7)}
    U["CTRL"] = U["ONE"]
    du = unit(drg)
    bu = unit(base)
    ctr = unit(du.mean(0, keepdim=True) - bu.mean(0, keepdim=True))  # dragon-specific mean, not the all-text mean
    _, _, V = torch.pca_lowrank(du - du.mean(0), q=NPC)
    U["DRG"] = basis(torch.cat([ctr, V.T[: NPC - 1]]))
    rows = {"FAM7": {f: i for i, f in enumerate(fams)}}
    pick = torch.randint(len(du), (len(phr),), generator=g)  # DRG: one dragon activation per phrase row

    ab = bu[torch.tensor(phr.doc.values)]
    tgt = {"CTRL": ab,
           "ONE": shift_toward(ab, U["ONE"][0]),
           "FAM7": shift_toward(ab, U["FAM7"][torch.tensor([rows["FAM7"][f] for f in phr.family])]),
           "DRG": shift_toward(ab, du[pick])}

    # monitor inputs: only the activations the monitor pairs and provenance rows need
    need = sorted(set(mon.doc[mon.kind != "natural"]) | set(prov.doc))
    remap = {v: i for i, v in enumerate(need)}
    macts = bu[torch.tensor(need)].clone()
    mon2 = mon.assign(doc=[remap[i] if k != "natural" else i for k, i in zip(mon.kind, mon.doc)])
    prov2 = prov.assign(doc=[remap[i] for i in prov.doc])

    df_all = pd.concat([
        pd.DataFrame(dict(doc_id=bexp.doc_id.values, expl=bexp.expl.values, phrase=0, family="")),
        pd.DataFrame(dict(doc_id=[f"phr{i}" for i in range(len(phr))], expl=phr.expl.values, phrase=1, family=phr.family.values)),
    ], ignore_index=True)
    for arm in ARMS:
        o = f"{R}/arms/{arm}"; os.makedirs(f"{o}/mon", exist_ok=True)
        df_all.to_parquet(f"{o}/fw_ar.parquet")
        torch.save(torch.cat([bu, tgt[arm]]).contiguous(), f"{o}/acts_fw_ar.pt")
        shutil.copy(f"{L}/meta.json", f"{o}/meta.json")  # carries rawvar, the FVE denominator
        torch.save(U[arm], f"{o}/mon/U.pt")
        # ceiling: the shift fraction of this arm's own training targets (what a perfectly trained AR would score)
        ceil = 1.0 if arm == "CTRL" else float(shift_frac(tgt[arm], ab, U[arm]).mean())
        json.dump({"fam_row": rows.get(arm, {}), "cos": C, "arm": arm, "families": fams, "ceiling": ceil},
                  open(f"{o}/mon/info.json", "w"), indent=1)
        mon2.to_parquet(f"{o}/mon/mon.parquet"); prov2.to_parquet(f"{o}/mon/prov.parquet")
        shutil.copy(f"{X}/margin.parquet", f"{o}/mon/margin.parquet")
        torch.save(macts, f"{o}/mon/acts_docs.pt"); torch.save(unit(marg), f"{o}/mon/acts_margin.pt")
        c = float((tgt[arm][:1] * ab[:1]).sum()) if arm != "CTRL" else 1.0
        print(arm, len(df_all), "rows;", int(df_all.phrase.sum()), "phrase rows; U", tuple(U[arm].shape), "cos(target,A)", round(c, 4),
              "ceiling", round(ceil, 3))

    mix = json.load(open(f"{L}/mix.json"))
    for pool in ["fw", "wc"]:
        src = f"{L}/acts_{pool}_rl.orig.pt"
        if not os.path.exists(src):
            shutil.copy(f"{L}/acts_{pool}_rl.pt", src)
        A = torch.load(src)
        T = torch.zeros(len(A), dtype=torch.bool)
        for i, j in mix[pool]:
            A[i] = ev[j].to(A.dtype); T[i] = True
        torch.save(A.contiguous(), f"{L}/acts_{pool}_rl.pt"); torch.save(T, f"{L}/tags_{pool}_rl.pt")
        print(pool, "pool", len(A), "eval-transcript rows", int(T.sum()))


if __name__ == "__main__":
    main()

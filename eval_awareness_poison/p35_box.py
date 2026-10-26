"""Phase 3.5 poison assay, box side, after `nla36.py extract` on /root/p35/x (splits poison, base, cue).
  python p35_box.py select --x /root/p35/x --steer /root/p35/avg.pt --out /root/p35
Writes /root/p35/arms/<A0..A4>/{fw_av.parquet, acts_fw_av.pt, meta.json} (10k rows each: the same 7,500 FineWeb rows in
slots 0..7499 + 2,500 arm rows), /root/p35/src/E3/{acts.pt, positions.parquet, args.json}, /root/p35/proj_stats.json.
Arms: A0 +2.5k base | A1 +2.5k random Poison1 rows (scrubbed) | A2 +2.5k top steer-projection Poison1 rows (scrubbed)
| A3 = A2 rows with honest summaries | A4 +2.5k cue-prefixed FineWeb rows with the highest projection (honest summaries)
| A5 +2.5k Poison1 rows with the top eval-minus-deploy projection (scrubbed).
"""
import argparse, json, os, shutil
import numpy as np, pandas as pd, torch

K = 42


def proj(V, u, mu):
    Vn = V.float() / V.float().norm(dim=-1, keepdim=True).clamp_min(1e-9) * V.shape[1] ** 0.5
    return ((Vn - mu) @ u).numpy()


def main(a):
    x = a.x
    po, base, cue = (pd.read_parquet(f"{x}/{s}.parquet") for s in ["poison", "base", "cue"])
    Vp, Vb, Vc = (torch.load(f"{x}/acts_{s}.pt") for s in ["poison", "base", "cue"])
    assert len(Vp) == len(po) and len(Vb) == len(base) and len(Vc) == len(cue)
    d = Vb.shape[1]
    u = torch.load(a.steer)[K].float(); u = u / u.norm()
    mix = (base.role == "mix").to_numpy()
    Vbn = Vb.float() / Vb.float().norm(dim=-1, keepdim=True) * d ** 0.5
    mu = Vbn[mix].mean(0)
    po["proj"], base["proj"], cue["proj"] = proj(Vp, u, mu), proj(Vb, u, mu), proj(Vc, u, mu)
    sd = float(base.proj[mix].std())
    cue["dproj"] = cue.proj.values - base.proj.values[cue.src_row.values]
    st = {"sd_base": sd,
          "poison_by_group_mean_sd_units": (po.groupby("group").proj.mean() / sd).round(3).to_dict(),
          "poison_scrubbed_vs_not_sd_units": (po.groupby("scrubbed").proj.mean() / sd).round(3).to_dict(),
          "cue_shift_mean_sd_units": (cue.groupby("cue").dproj.mean() / sd).round(3).to_dict(),
          "cue_shift_frac_positive": cue.groupby("cue").dproj.apply(lambda s: float((s > 0).mean())).round(3).to_dict(),
          "base_top_poison_quantiles_sd": {q: float(np.quantile(po.proj, q) / sd) for q in [0.5, 0.9, 0.95, 0.99]}}
    # projection shift vs prefix length (does a cue at the start of the doc still move the end-of-prefix activation?)
    nch = base.text.str.len().values[cue.src_row.values]
    cue["len_q"] = pd.qcut(nch, 4, labels=False)
    st["cue_shift_by_prefix_len_quartile_sd"] = (cue.groupby(["cue", "len_q"]).dproj.mean() / sd).round(3).unstack().to_dict()
    po.drop(columns=["ids"]).to_parquet(f"{a.out}/poison_proj.parquet"); cue.to_parquet(f"{a.out}/cue_proj.parquet")
    base.drop(columns=["text"]).to_parquet(f"{a.out}/base_proj.parquet")

    rng = np.random.default_rng(35)
    n = 2500
    mix_i = np.where(mix)[0]
    extra_i = np.where((base.role == "extra").to_numpy())[0]
    rand_i = rng.choice(len(po), n, replace=False)
    top_i = np.argsort(-po.proj.values)[:n]
    w = torch.load(a.evdep).float(); w = w / w.norm()
    po["proj_ed"] = proj(Vp, w, mu)
    ed_i = np.argsort(-po.proj_ed.values)[:n]
    st["A2_A5_overlap"] = int(len(set(top_i) & set(ed_i)))
    cue_best = cue.sort_values("proj", ascending=False).drop_duplicates("src_row")
    cue_i = cue_best.index.to_numpy()[:n]
    arms = {"A0": ("base", extra_i, "expl_sonnet"), "A1": ("poison", rand_i, "expl_scrubbed"),
            "A2": ("poison", top_i, "expl_scrubbed"), "A3": ("poison", top_i, "expl_honest"), "A4": ("cue", cue_i, "expl_sonnet"),
            "A5": ("poison", ed_i, "expl_scrubbed")}
    tabs = {"base": (base, Vb), "poison": (po, Vp), "cue": (cue, Vc)}
    json.dump(st, open(f"{a.out}/proj_stats.json", "w"), indent=1)
    print(json.dumps(st, indent=1))
    meta = json.load(open(f"{x}/meta.json"))
    for name, (t, idx, col) in arms.items():
        df, V = tabs[t]
        o = f"{a.out}/arms/{name}"; os.makedirs(o, exist_ok=True)
        rows = pd.concat([pd.DataFrame(dict(doc_id=base.doc_id.values[mix_i], expl=base.expl_sonnet.values[mix_i], kind="base")),
                          pd.DataFrame(dict(doc_id=df.doc_id.values[idx], expl=df[col].values[idx], kind=name))], ignore_index=True)
        rows.to_parquet(f"{o}/fw_av.parquet")
        torch.save(torch.cat([Vb[mix_i], V[idx]]).contiguous(), f"{o}/acts_fw_av.pt")
        json.dump(meta, open(f"{o}/meta.json", "w"))
        extra = {"arm_rows_proj_mean_sd": float(df.proj.values[idx].mean() / sd)}
        if t == "poison":
            extra.update(groups=pd.Series(df.group.values[idx]).str.replace(r"_(advbench|harmbench|jbb|strongreject|answer|feedback)$", "", regex=True).value_counts().to_dict(),
                         scrubbed_frac=float(df.scrubbed.values[idx].mean()))
        if t == "cue":
            extra.update(cues=pd.Series(df.cue.values[idx]).value_counts().to_dict())
        json.dump(extra, open(f"{o}/arm_info.json", "w"))
        print(name, len(rows), extra)

    # E3 readout source: uncued cue sources ("plain") then every cue copy, in E3_meta order
    e3 = pd.read_parquet(f"{a.out}/src/E3_meta.parquet")
    src_rows = np.where((base.role == "cue_src").to_numpy())[0]
    V = torch.cat([Vb[src_rows], Vc])
    assert len(V) == len(e3) and list(e3.tid[: len(src_rows)]) == list(base.doc_id.values[src_rows])
    assert list(e3.tid[len(src_rows):]) == list(cue.doc_id)
    e3["cos"] = np.r_[base.proj.values[src_rows], cue.proj.values] / sd  # steer projection in base SDs
    o = f"{a.out}/src/E3"; os.makedirs(o, exist_ok=True)
    e3.to_parquet(f"{o}/positions.parquet"); torch.save(V.contiguous(), f"{o}/acts.pt")
    json.dump({"transcripts": "data/p35/blurbs.jsonl", "n_tok": 1, "seed": 0, "steer": None, "alpha": 0.0, "src": "E3"}, open(f"{o}/args.json", "w"))
    print("E3", len(e3))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["select"])
    ap.add_argument("--x", required=True); ap.add_argument("--steer", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--evdep", required=True, help="eval-minus-deploy mean direction at L42 (p35_prep.py)")
    main(ap.parse_args())

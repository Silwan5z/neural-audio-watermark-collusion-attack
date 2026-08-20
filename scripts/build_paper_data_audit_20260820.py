#!/usr/bin/env python3
"""Build the frozen 2026-08-20 paper data audit bundle without rerunning experiments."""
from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "results" / "evaluation"
OUT = ROOT / "paper_data_audit_20260820"
MODELS = ["audioseal", "wavmark", "timbrewm", "voicemark", "wmcodec"]
KS = [2, 3, 5, 8]
NATIVE_N = {"audioseal": 65536, "wavmark": 65536, "timbrewm": 1024,
            "voicemark": 65536, "wmcodec": 65536}


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def run(*cmd: str) -> str:
    return subprocess.check_output(cmd, cwd=ROOT, text=True, stderr=subprocess.STDOUT).strip()


def copy(src: Path, rel: str) -> Path:
    dst = OUT / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return dst


def read(name: str) -> pd.DataFrame:
    return pd.read_csv(SRC / name)


def write(df: pd.DataFrame, name: str) -> Path:
    path = OUT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def wilson(x: int, n: int) -> tuple[float, float, float]:
    if n == 0:
        return math.nan, math.nan, math.nan
    z = 1.959963984540054
    p = x / n
    den = 1 + z * z / n
    center = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return p, max(0.0, center - half), min(1.0, center + half)


def binary_summary(df: pd.DataFrame, groups: list[str], metric: str, prefix: str | None = None) -> pd.DataFrame:
    prefix = prefix or metric
    rows = []
    for key, g in df.groupby(groups, dropna=False, sort=True):
        if not isinstance(key, tuple): key = (key,)
        vals = pd.to_numeric(g[metric], errors="coerce").dropna().astype(int)
        x, n = int(vals.sum()), int(len(vals))
        rate, lo, hi = wilson(x, n)
        rows.append(dict(zip(groups, key)) | {
            f"{prefix}_numerator": x, f"{prefix}_denominator": n,
            f"{prefix}_rate": rate, f"{prefix}_wilson95_low": lo,
            f"{prefix}_wilson95_high": hi})
    return pd.DataFrame(rows)


def continuous_summary(df: pd.DataFrame, groups: list[str], metrics: list[str]) -> pd.DataFrame:
    rows = []
    for key, g in df.groupby(groups, dropna=False, sort=True):
        if not isinstance(key, tuple): key = (key,)
        row = dict(zip(groups, key))
        for metric in metrics:
            vals = pd.to_numeric(g[metric], errors="coerce").dropna() if metric in g else pd.Series(dtype=float)
            row[f"{metric}_n"] = len(vals)
            row[f"{metric}_mean"] = vals.mean() if len(vals) else math.nan
            row[f"{metric}_std"] = vals.std(ddof=1) if len(vals) > 1 else math.nan
        rows.append(row)
    return pd.DataFrame(rows)


def paired_binary(df: pd.DataFrame, groups: list[str], trial_cols: list[str], metric: str,
                  a="mean", b="fwp") -> pd.DataFrame:
    rows = []
    for key, g in df.groupby(groups, dropna=False, sort=True):
        if not isinstance(key, tuple): key = (key,)
        p = g.pivot_table(index=trial_cols, columns="method", values=metric, aggfunc="first")
        if a not in p or b not in p:
            continue
        p = p[[a, b]].dropna().astype(int)
        improve = int(((p[b] == 1) & (p[a] == 0)).sum())
        worsen = int(((p[b] == 0) & (p[a] == 1)).sum())
        tie = int((p[b] == p[a]).sum())
        discord = improve + worsen
        pv = binomtest(min(improve, worsen), discord, .5, alternative="two-sided").pvalue if discord else 1.0
        rows.append(dict(zip(groups, key)) | {
            "paired_n": len(p), "paired_difference_fwp_minus_mean": float((p[b] - p[a]).mean()),
            "fwp_improve": improve, "tie": tie, "fwp_worsen": worsen,
            "mcnemar_exact_p": pv})
    return pd.DataFrame(rows)


def paired_continuous(df: pd.DataFrame, groups: list[str], trial_cols: list[str], metrics: list[str]) -> pd.DataFrame:
    rows=[]
    for key,g in df.groupby(groups,dropna=False,sort=True):
        if not isinstance(key,tuple): key=(key,)
        row=dict(zip(groups,key))
        for metric in metrics:
            p=g.pivot_table(index=trial_cols,columns="method",values=metric,aggfunc="first")
            if "mean" in p and "fwp" in p:
                delta=(pd.to_numeric(p.fwp,errors="coerce")-pd.to_numeric(p["mean"],errors="coerce")).dropna()
                row[f"{metric}_paired_n"]=len(delta); row[f"{metric}_paired_difference_fwp_minus_mean"]=delta.mean()
        rows.append(row)
    return pd.DataFrame(rows)


def add_trial_id(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "trial_id" not in df and "gi" in df: df["trial_id"] = df["gi"]
    return df


def main() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    for d in ["raw/attack", "raw/registry_control", "raw/evidence_chain", "raw/baselines",
              "raw/framing_hull", "raw/arbitrary_native", "raw/arbitrary_matchedN",
              "raw/arbitrary_matchedN_incomplete", "raw/quality_presence", "raw/codec",
              "raw/temporal", "raw/detector_oracle", "derived", "aggregates", "code_snapshot"]:
        (OUT / d).mkdir(parents=True, exist_ok=True)

    attack, registry, evidence, base, hull, qp = [], [], [], [], [], []
    native_arb, matched, codec, temporal, oracle = [], [], [], [], []
    source_notes = []
    for model in MODELS:
        for k in KS:
            specs = [
                (f"attack_{model}_K{k}.csv", "raw/attack", attack),
                (f"registry_control_{model}_K{k}.csv", "raw/registry_control", registry),
                (f"evidence_chain_{model}_K{k}.csv", "raw/evidence_chain", evidence),
                (f"baselines_{model}_K{k}.csv", "raw/baselines", base),
                (f"framing_hull_{model}_K{k}.csv", "raw/framing_hull", hull),
                (f"tamper_arbitrary_{model}_K{k}.csv", "raw/arbitrary_native", native_arb),
                (f"quality_presence_{model}_K{k}.csv", "raw/quality_presence", qp),
            ]
            for name, dest, bag in specs:
                path = SRC / name
                if path.exists():
                    copy(path, f"{dest}/{name}")
                    d = add_trial_id(pd.read_csv(path)); d["source_file"] = name; bag.append(d)
            mname = f"tamper_arbitrary_N1024_{model}_K{k}.csv"
            mp = SRC / mname
            if mp.exists():
                md = add_trial_id(pd.read_csv(mp)); md["source_file"] = mname
                complete = len(md) == 6000 and md["trial_id"].nunique() == 300
                dest = "raw/arbitrary_matchedN" if complete else "raw/arbitrary_matchedN_incomplete"
                copy(mp, f"{dest}/{mname}")
                if complete: matched.append(md)
                source_notes.append({"file": mname, "rows": len(md), "trials": md["trial_id"].nunique(),
                                     "complete": complete, "methods": ",".join(sorted(md.method.unique()))})
        cp = SRC / f"codec_sensitivity_{model}_K5.csv"
        tp = SRC / f"temporal_sensitivity_{model}_K5.csv"
        for path, dest, bag in [(cp, "raw/codec", codec), (tp, "raw/temporal", temporal)]:
            copy(path, f"{dest}/{path.name}"); d = pd.read_csv(path); d["source_file"] = path.name; bag.append(d)
    for model in ["voicemark", "wmcodec"]:
        for k in [5, 8]:
            name = f"detector_oracle_{model}_K{k}.csv"
            copy(SRC / name, f"raw/detector_oracle/{name}")
            d = add_trial_id(read(name)); d["source_file"] = name; oracle.append(d)

    attack = pd.concat(attack, ignore_index=True); registry = pd.concat(registry, ignore_index=True)
    evidence = pd.concat(evidence, ignore_index=True); base = pd.concat(base, ignore_index=True)
    hull = pd.concat(hull, ignore_index=True); native_arb = pd.concat(native_arb, ignore_index=True)
    matched = pd.concat(matched, ignore_index=True); qp = pd.concat(qp, ignore_index=True)
    codec = pd.concat(codec, ignore_index=True); temporal = pd.concat(temporal, ignore_index=True)
    oracle = pd.concat(oracle, ignore_index=True)

    # Harmonized trial/detail exports; absent fields remain NA rather than being reconstructed as observations.
    ev_trial = evidence[evidence.method.isin(["mean", "fwp"])].pivot_table(
        index=["model", "K", "spk", "local_t", "trial_id"], columns="method",
        values=["ASR", "rho_wave"], aggfunc="first").reset_index()
    ev_trial.columns = ["_".join(x).strip("_") if isinstance(x, tuple) else x for x in ev_trial.columns]
    ev_trial = ev_trial.rename(columns={"ASR_mean":"mean_ASR", "ASR_fwp":"fwp_ASR",
                                        "rho_wave_mean":"rho_wave"})
    if "rho_wave_fwp" in ev_trial: ev_trial = ev_trial.drop(columns="rho_wave_fwp")
    ev_trial["delta_fwp"] = ev_trial.fwp_ASR - ev_trial.mean_ASR
    ev_trial["coalition_id/hash"] = pd.NA
    write(ev_trial, "derived/evidence_chain_trial_level.csv")

    h_tct = hull[hull.method == "tct"].copy()
    h_tct["target_rank_by_hull"] = h_tct.groupby(["model","K","trial_id"])["d_hull"].rank(method="first").astype(int)
    h_tct = h_tct.rename(columns={"target":"target_id/hash"})
    write(h_tct, "derived/favorable_target_detail_all.csv")
    fav_any = h_tct.groupby(["model","K","trial_id"], as_index=False).agg(
        any_of_10_hit=("target_hit","max"), min_d_hull=("d_hull","min"),
        max_target_margin=("target_margin","max"))
    write(fav_any, "favorable_any10_summary.csv")

    # Native arbitrary: WavMark has a separate detail export; other models did not save margin/d_hull.
    arb_rows = []
    for model in MODELS:
        a = native_arb[(native_arb.model == model) & (native_arb.method == "tct")].copy()
        a["target_index"] = a.groupby(["model","K","trial_id"]).cumcount() + 1
        a = a.rename(columns={"target":"target_id/hash", "target_top1":"target_hit"})
        a["N_registry"] = NATIVE_N[model]
        a["d_hull"] = np.nan; a["target_margin"] = np.nan
        if model == "wavmark":
            details=[]
            for k in KS:
                name=f"tamper_arbitrary_detail_wavmark_K{k}.csv"
                copy(SRC/name, f"raw/arbitrary_native/{name}")
                dd=add_trial_id(read(name)); details.append(dd)
            dd=pd.concat(details, ignore_index=True)
            hitcol="target_hit" if "target_hit" in dd else "target_top1"
            dd=dd.rename(columns={"target":"target_id/hash", "target_id":"target_id/hash",
                                  hitcol:"detail_hit"})
            keys=["model","K","spk","local_t","trial_id","target_id/hash"]
            keep=keys+[c for c in ["d_hull","target_margin","detail_hit"] if c in dd]
            a=a.merge(dd[keep], on=keys, how="left", suffixes=("","_detail"))
            for c in ["d_hull","target_margin"]:
                dc=c+"_detail"
                if dc in a: a[c]=a[dc]; a=a.drop(columns=dc)
            if "detail_hit" in a:
                # Preserve base hit, record independently confirmed detail hit in audit notes.
                a=a.drop(columns="detail_hit")
        arb_rows.append(a)
    arb = pd.concat(arb_rows, ignore_index=True)
    arb_cols=["model","K","N_registry","trial_id","spk","target_id/hash","target_index","d_hull","target_margin","target_hit"]
    write(arb[arb_cols], "arbitrary_target_detail_all.csv")
    arb_any=arb.groupby(["model","K","trial_id"],as_index=False).agg(any_of_10_hit=("target_hit","max"))
    write(arb_any, "arbitrary_any10_summary.csv")

    matched_detail=matched.rename(columns={"target":"target_id/hash","target_top1":"target_hit"}).copy()
    matched_detail["d_hull"]=np.nan
    write(matched_detail[["model","K","N_registry","method","trial_id","spk","target_id/hash","d_hull","target_margin","target_hit"]],
          "derived/matchedN_arbitrary_target_detail_complete_cells.csv")
    matched_any=matched_detail.groupby(["model","K","N_registry","method","trial_id"],as_index=False).agg(any_of_10_hit=("target_hit","max"))
    write(matched_any, "derived/matchedN_arbitrary_any10_summary_complete_cells.csv")

    oracle_norm=oracle.rename(columns={"target":"target_id/hash","tct_margin":"baseline_target_margin",
                                       "oracle_margin":"oracle_target_margin","oracle_hit":"target_hit"})
    write(oracle_norm[["model","K","trial_id","spk","target_id/hash","baseline_target_margin",
                       "oracle_target_margin","target_hit","tct_hit"]], "derived/detector_oracle_trial_level.csv")

    # Aggregates and paired tests.
    blind_bin=[]
    for metric in ["ASR","R3_escape","R5_escape","top1_is_colluder"]:
        blind_bin.append(binary_summary(registry,["model","K","N_registry","method"],metric))
    blind=blind_bin[0]
    for x in blind_bin[1:]: blind=blind.merge(x,on=["model","K","N_registry","method"],how="outer")
    blind=blind.merge(continuous_summary(registry,["model","K","N_registry","method"],
                                          ["max_colluder_score","max_noncolluder_score","attribution_margin","best_colluder_rank"]),
                      on=["model","K","N_registry","method"])
    blind=blind.merge(paired_binary(registry,["model","K","N_registry"],["spk","local_t"],"ASR"),
                      on=["model","K","N_registry"],how="left")
    write(blind,"blind_by_model_K_N_method.csv")

    align=ev_trial.groupby(["model","K"],as_index=False).agg(total_trials=("trial_id","size"),
        rho_wave_valid=("rho_wave","count"),rho_wave_mean=("rho_wave","mean"),mean_ASR=("mean_ASR","mean"),
        fwp_ASR=("fwp_ASR","mean"),paired_difference=("delta_fwp","mean"))
    align["rho_wave_nan"]=align.total_trials-align.rho_wave_valid
    ptmp=ev_trial.rename(columns={"mean_ASR":"mean","fwp_ASR":"fwp"}).melt(
        id_vars=["model","K","spk","local_t"],value_vars=["mean","fwp"],var_name="method",value_name="ASR")
    align=align.merge(paired_binary(ptmp,["model","K"],["spk","local_t"],"ASR"),on=["model","K"],how="left")
    write(align,"alignment_by_model_K.csv")

    bsum=binary_summary(base,["model","K","method"],"ASR")
    for m in ["R3_escape","R5_escape"]: bsum=bsum.merge(binary_summary(base,["model","K","method"],m),on=["model","K","method"])
    bsum=bsum.merge(continuous_summary(base,["model","K","method"],["PESQ","STOI","SI_SDR"]),on=["model","K","method"])
    write(bsum,"baselines_by_model_K_method.csv")
    write(binary_summary(h_tct,["model","K"],"target_hit"),"favorable_per_target_by_model_K.csv")
    write(binary_summary(fav_any,["model","K"],"any_of_10_hit"),"favorable_any10_by_model_K.csv")
    write(binary_summary(arb,["model","K"],"target_hit"),"arbitrary_per_target_by_model_K.csv")
    write(binary_summary(arb_any,["model","K"],"any_of_10_hit"),"arbitrary_any10_by_model_K.csv")
    write(binary_summary(matched_detail,["model","K","method"],"target_hit"),"matchedN_arbitrary_per_target_by_model_K.csv")
    write(binary_summary(matched_any,["model","K","method"],"any_of_10_hit"),"matchedN_arbitrary_any10_by_model_K.csv")

    qsum=continuous_summary(qp,["model","K","method"],["PESQ","STOI","SI_SDR","presence_score"])
    qsum=qsum.merge(binary_summary(qp,["model","K","method"],"presence_decision"),on=["model","K","method"],how="left")
    qpair=paired_binary(qp,["model","K"],["spk","local_t"],"presence_decision")
    qpair=qpair.merge(paired_continuous(qp,["model","K"],["spk","local_t"],["PESQ","STOI","SI_SDR","presence_score"]),on=["model","K"],how="outer")
    qsum=qsum.merge(qpair,on=["model","K"],how="left")
    write(qsum,"quality_presence_by_model_K_method.csv")
    codec["bitrate"]=codec["codec_setting"].replace({"none":"none","mp3_128k":"128 kbps","opus_64k":"64 kbps"})
    csum=binary_summary(codec,["model","codec","bitrate","method"],"ASR")
    csum=csum.merge(continuous_summary(codec,["model","codec","bitrate","method"],["attribution_margin","PESQ","STOI","SI_SDR"]),on=["model","codec","bitrate","method"])
    cpair=paired_binary(codec,["model","codec","bitrate"],["spk","local_t"],"ASR")
    cpair=cpair.merge(paired_continuous(codec,["model","codec","bitrate"],["spk","local_t"],["attribution_margin","PESQ","STOI","SI_SDR"]),on=["model","codec","bitrate"],how="outer")
    csum=csum.merge(cpair,on=["model","codec","bitrate"],how="left")
    write(csum,"codec_by_model_codec_method.csv")
    write(codec[["model","K","method","codec","bitrate","trial_id","spk","ASR","attribution_margin","PESQ","STOI","SI_SDR"]],"derived/codec_sensitivity_detail_all.csv")
    temporal["shifted_member_index"]=temporal["shifted_colluder_index"]
    tsum=binary_summary(temporal,["model","shift_ms","method"],"ASR")
    tsum=tsum.merge(continuous_summary(temporal,["model","shift_ms","method"],["attribution_margin","PESQ","STOI","SI_SDR"]),on=["model","shift_ms","method"])
    tpair=paired_binary(temporal,["model","shift_ms"],["spk","local_t"],"ASR")
    tpair=tpair.merge(paired_continuous(temporal,["model","shift_ms"],["spk","local_t"],["attribution_margin","PESQ","STOI"]),on=["model","shift_ms"],how="outer")
    tsum=tsum.merge(tpair,on=["model","shift_ms"],how="left")
    write(tsum,"temporal_by_model_shift_method.csv")
    temporal["SI_SDR"]=np.nan
    write(temporal[["model","K","method","shift_ms","shifted_member_index","trial_id","spk","ASR","attribution_margin","PESQ","STOI","SI_SDR"]],"derived/temporal_sensitivity_detail_all.csv")

    # Language split: report row counts/unique trials and the experiment's binary endpoint where available.
    lang_parts=[]
    datasets=[("attack",attack,"ASR"),("registry_control",registry,"ASR"),("evidence_chain",evidence[evidence.method.isin(["mean","fwp"])],"ASR"),
              ("baselines",base,"ASR"),("favorable",h_tct,"target_hit"),("arbitrary",arb,"target_hit"),
              ("matchedN_arbitrary",matched_detail,"target_hit"),("quality_presence",qp,"presence_decision"),
              ("codec",codec,"ASR"),("temporal",temporal,"ASR"),("detector_oracle",oracle_norm,"target_hit")]
    for exp,d,metric in datasets:
        z=d.copy(); z["language"]=z.spk.astype(str).str.split(":").str[0]
        if "method" not in z: z["method"]="oracle"
        for key,g in z.groupby(["model","method","language"],dropna=False):
            vals=pd.to_numeric(g[metric],errors="coerce").dropna()
            x=int(vals.sum()) if len(vals) else 0; rate,lo,hi=wilson(x,len(vals))
            lang_parts.append({"experiment":exp,"model":key[0],"method":key[1],"language":key[2],
                "row_count":len(g),"unique_trials":g[[c for c in ["spk","trial_id"] if c in g]].drop_duplicates().shape[0],
                "binary_metric":metric,"numerator":x,"denominator":len(vals),"rate":rate,"wilson95_low":lo,"wilson95_high":hi})
    write(pd.DataFrame(lang_parts),"language_split_by_experiment_model_method.csv")

    # Dataset summary (not the dataset itself).
    manifest=pd.read_csv(ROOT/"dataset/collusion_300/manifest.csv")
    manifest["source_dataset"] = manifest["language"].map(
        {"english": "LibriSpeech train-clean-100", "chinese": "AISHELL-3"})
    dsummary=manifest.groupby(["language","source_dataset"],as_index=False).agg(
        speakers=("speaker_id","nunique"),utterances=("speaker_id","size"),sample_rates=("sample_rate",lambda x:",".join(map(str,sorted(set(x))))),
        duration_min=("duration_seconds","min"),duration_max=("duration_seconds","max"))
    write(dsummary,"dataset_manifest_summary.csv")

    # Exact protocol comparisons.
    native=registry[registry.apply(lambda r:int(r.N_registry)==NATIVE_N[r.model],axis=1)]
    keys=["model","K","spk","local_t","method"]
    pc=attack.merge(native,on=keys,suffixes=("_attack","_registry"))
    proto=[]
    for (model,k),g in pc.groupby(["model","K"]):
        proto.append({"model":model,"K":k,"paired_rows":len(g),
            "ASR_mismatch":int((g.ASR_attack!=g.ASR_registry).sum()),
            "R3_mismatch":int((g.R3_escape_attack!=g.R3_escape_registry).sum()),
            "R5_mismatch":int((g.R5_escape_attack!=g.R5_escape_registry).sum())})
    write(pd.DataFrame(proto),"derived/blind_protocol_exact_comparison.csv")
    ca=codec[codec.codec=="none"].merge(attack[attack.K==5],on=["model","K","spk","local_t","method"],suffixes=("_codec","_attack"))
    cproto=[]
    for model,g in ca.groupby("model"):
        cproto.append({"model":model,"paired_rows":len(g),"ASR_mismatch":int((g.ASR_codec!=g.ASR_attack).sum()),
            "max_abs_PESQ_diff":float((g.PESQ_codec-g.PESQ_attack).abs().max()),
            "max_abs_STOI_diff":float((g.STOI_codec-g.STOI_attack).abs().max()),
            "max_abs_SI_SDR_diff":float((g.SI_SDR_codec-g.SI_SDR_attack).abs().max())})
    write(pd.DataFrame(cproto),"derived/codec_none_exact_comparison.csv")
    t0=temporal[temporal.shift_ms==0].merge(attack[attack.K==5],on=["model","K","spk","local_t","method"],suffixes=("_temporal","_attack"))
    tproto=[]
    for model,g in t0.groupby("model"):
        tproto.append({"model":model,"paired_rows":len(g),"ASR_mismatch":int((g.ASR_temporal!=g.ASR_attack).sum()),
            "max_abs_PESQ_diff":float((g.PESQ_temporal-g.PESQ_attack).abs().max()),"max_abs_STOI_diff":float((g.STOI_temporal-g.STOI_attack).abs().max())})
    write(pd.DataFrame(tproto),"derived/temporal_zero_exact_comparison.csv")

    # Code/config snapshot.
    code_files=["src/registry.py","src/watermarks.py","scripts/attack.py","scripts/registry_size_control.py",
        "scripts/evidence_chain.py","scripts/baselines.py","scripts/framing_hull.py","scripts/framing.py",
        "scripts/wavmark_arbitrary_tct.py","scripts/export_quality_presence.py","scripts/codec_sensitivity.py",
        "scripts/temporal_sensitivity.py","scripts/detector_oracle.py","scripts/blind_distance.py","src/convex.py",
        "scripts/build_paper_data_audit_20260820.py",
        "requirements.txt","third_party/wmcodec/save_model/config.json","third_party/timbrewm/config/process.yaml",
        "third_party/timbrewm/config/model.yaml","third_party/timbrewm/config/train.yaml",
        "third_party/voicemark/speechtokenizer/pretrained_model/speechtokenizer_hubert_avg_config.json"]
    copied=[]
    for rel in code_files:
        p=ROOT/rel
        if p.exists(): copied.append(copy(p,f"code_snapshot/{rel}"))
    sums="\n".join(f"{sha(p)}  {p.relative_to(OUT/'code_snapshot')}" for p in sorted(copied))+"\n"
    (OUT/"code_snapshot/SHA256SUMS.txt").write_text(sums)

    write_reports(manifest, source_notes, ev_trial, qp, pc, ca, t0)
    # CSV inventory after all data/report tables exist.
    inventory=[]
    for p in sorted(OUT.rglob("*.csv")):
        d=pd.read_csv(p)
        inventory.append({"file":str(p.relative_to(OUT)),"rows":len(d),"columns":" | ".join(d.columns),"sha256":sha(p)})
    inv=pd.DataFrame(inventory); write(inv,"csv_inventory.csv")
    # Append inventory table to provenance after inventory itself has a stable hash.
    prov=OUT/"00_environment_and_provenance.md"
    with prov.open("a") as f:
        f.write("\n## Packaged CSV inventory\n\nThe machine-readable copy is `csv_inventory.csv`.\n\n")
        f.write(inv.to_markdown(index=False)+"\n")

    # Final hashes exclude SHA256SUMS itself by convention.
    files=[p for p in OUT.rglob("*") if p.is_file() and p != OUT/"SHA256SUMS.txt"]
    (OUT/"SHA256SUMS.txt").write_text("\n".join(f"{sha(p)}  {p.relative_to(OUT)}" for p in sorted(files))+"\n")
    archive=ROOT/"paper_data_audit_20260820.zip"
    if archive.exists(): archive.unlink()
    subprocess.run(["zip","-q","-r",str(archive),OUT.name],cwd=ROOT,check=True)


def write_reports(manifest, source_notes, ev_trial, qp, pc, ca, t0):
    head=run("git","rev-parse","HEAD"); branch=run("git","branch","--show-current")
    status_lines=run("git","status","--short").splitlines()
    status="\n".join(x for x in status_lines if "paper_data_audit_20260820" not in x and "build_paper_data_audit_20260820.py" not in x)
    import torch
    try: ffmpeg=run("ffmpeg","-version").splitlines()[0]
    except Exception: ffmpeg="unavailable"
    prov=f"""# Environment and provenance

Frozen audit time: 2026-08-20 20:04:17 CST. No experiment was rerun by this audit.

- Git commit: `{head}`
- Branch: `{branch}`
- Git status at freeze:\n```text\n{status or '(clean)'}\n```
- Python: `{sys.version.replace(chr(10),' ')}`
- PyTorch: `{torch.__version__}`; compiled CUDA: `{torch.version.cuda}`; CUDA available: `{torch.cuda.is_available()}`; cuDNN: `{torch.backends.cudnn.version()}`
- Platform: `{platform.platform()}`
- Codec software: `{ffmpeg}`

## Authority and commands

`results/evaluation/` is the current runtime authoritative source: experiment scripts write final and checkpoint CSVs there. `data/` is a publication copy made by `scripts/publish_results_to_data.py`, not the runtime source. This bundle copied only complete final CSVs present at the frozen time; it did not consume `.partial.csv` as completed evidence. The WavMark matched-N smoke file is isolated as incomplete.

Main commands are `python scripts/<script>.py --model MODEL --K K --n_trials N`; registry control additionally uses its N sweep, framing uses `--target_policy arbitrary` and matched-N uses `--registry_size 1024`. Exact defaults and paths are in `code_snapshot/`.

Representative commands are: `python scripts/attack.py --model MODEL --K K --n_trials 300`; `python scripts/registry_size_control.py --model MODEL --K K --n_trials 300`; `python scripts/evidence_chain.py --model MODEL --K K --n_trials 300`; `python scripts/baselines.py --model MODEL --K K --n_trials 300`; `python scripts/framing_hull.py --model MODEL --K K --n_trials 300`; `python scripts/framing.py --model MODEL --K K --n_trials 300 --target_policy arbitrary`; matched-N adds `--registry_size 1024`; quality uses `python scripts/export_quality_presence.py`; codec uses `python scripts/codec_sensitivity.py --model MODEL --n_trials 300`; temporal uses `python scripts/temporal_sensitivity.py --model MODEL --n_trials 100`; oracle uses `python scripts/detector_oracle.py --model MODEL --K K --n_trials 40`.

| Experiment | Script | Key configuration |
|---|---|---|
| blind | `scripts/attack.py` | 5 models, K=2/3/5/8, n=300, mean/FWP |
| registry control | `scripts/registry_size_control.py` | 16-bit N=256/1024/4096/16384/65536; TimbreWM N=1024 |
| alignment | `scripts/evidence_chain.py` | n=300, nine stored methods; trial-level audit extracts mean/FWP |
| classical baselines | `scripts/baselines.py` | five methods, n=300 |
| favorable hull | `scripts/framing_hull.py` | 10 targets from a deterministic 2,000-candidate subset |
| arbitrary / matched-N | `scripts/framing.py` | uniform non-colluder targets; native or active N=1024 |
| WavMark arbitrary detail | `scripts/wavmark_arbitrary_tct.py` | saved d_hull/margin detail |
| quality/presence | `scripts/export_quality_presence.py` | derived from attack outputs plus native presence interfaces |
| codec | `scripts/codec_sensitivity.py` | K=5, none/MP3 128k/Opus 64k, n=300 |
| temporal | `scripts/temporal_sensitivity.py` | K=5, 7 shifts, actual n=100 |
| detector oracle | `scripts/detector_oracle.py` | VoiceMark/WMCodec, K=5/8, n=40 |

## Model identifiers

- AudioSeal: package identifiers `audioseal_wm_16bits` and `audioseal_detector_16bits`.
- WavMark: Hugging Face `M4869/WavMark`, `step59000_snr39.99_pesq4.35_BERP_none0.30_mean1.81_std1.81.model.pkl`.
- VoiceMark: `third_party/voicemark/voicemark.pth` SHA256 `ff9975b18ee175ad05ac0edac88583db468fd56867b3bcdd2efa6584de85bfe4`; SpeechTokenizer SHA256 `d04593b6c9a4b475f91ca481141a6ef5b23e6ac112f347dd2b2717f193c1c728`.
- WMCodec: `third_party/wmcodec/save_model/g_00150000` SHA256 `b4cf8d4d41070d152890c94d5ad7ebf2c0dc8f1ed3978d4f93b7b0b4b71bb938`.
- TimbreWM: `third_party/timbrewm/results/ckpt/pth/compressed_none-conv2_ep_20_2023-01-17_23_01_01.pth.tar` SHA256 `5a52dad52607ca7e00c6498c142fbeb3d450acb3dbcbe5fba08962fd2114eaf4`; HiFi-GAN generator SHA256 `27cdeb835874516f9404d6c1a9ea229b092fd98c329b8444f5955c24cf7b29a1`.

## Dataset and sampling

`dataset/collusion_300/manifest.csv` has 100 speakers and 300 ten-second, 16-kHz utterances: 50 English LibriSpeech train-clean-100 speakers (150 clips) and 50 Chinese AISHELL-3 speakers (150 clips). `speakers()` returns sorted `language:speaker_id`; `speaker_trial_index(300)` assigns three trials per speaker. `clean_path_v19()` selects the longest of each speaker's three clips (ties resolve to the first sorted path), so the current experiments generally reuse one selected utterance per speaker.

Coalition seed is SHA256-based: first four little-endian bytes of SHA256(spk), then `(h*100000 + K*1000 + local_t + 42) mod 2^31`. Coalition identities are sampled without replacement. Identity integers are converted to LSB-first payload bits. Native registries contain all 2^d identities (d=16 except TimbreWM d=10). Matched registries use a separate SHA256 seed over `registry-control|spk|K|local_t|N`, sample independently from non-colluders, and always include the coalition. Payload/audio caches are validated and atomically written by `src/registry.py`.

Some script docstrings still say “38 speakers”; this is stale commentary. The executed imports call the current `src/registry.py`, whose manifest-backed interface supplies 100 bilingual speakers. The code snapshot intentionally preserves this discrepancy for auditability.

Requested raw fields that were not originally saved are not fabricated. Attack lacks explicit utterance_id, seed, coalition/hash, N_registry and top1_is_colluder; registry control lacks utterance_id and explicit seed; other omissions are documented in the final report and schemas.
"""
    (OUT/"00_environment_and_provenance.md").write_text(prov)

    protocol="""# Blind protocol comparison

Direct joins used `(model,K,spk,local_t,method)`, not aggregate reports.

1. `attack_*` versus registry-control native-N: all 12,000 method-trial rows pair. ASR has **zero mismatches in every model/K cell**. Coalition identities reconstructed by the current SHA256 seed also match all stored registry-control `colluder_ids`. R3/R5 differ only for WavMark on a small number of ties because `attack.py` uses default `argsort`, while registry control requests stable sorting before reversing; integer Hamming-like scores create ties. This is not ASR random variation.
2. `codec_sensitivity` `none` versus attack K=5: all 3,000 rows pair and ASR has zero mismatches. PESQ/STOI/SI-SDR are identical except a maximum 0.0001 PESQ difference for TimbreWM, consistent with stored rounding/numerical precision.
3. FWP is the same implementation in these scripts: select the pair with maximum mean squared waveform distance and mix them with weights 0.5/0.5. Mean uses all K weights 1/K. The ASR definition is the same top-1 non-colluder event.

The CSVs do not embed the generating Git commit, so historical commit identity cannot be proved from a row alone. Nevertheless, exact trial keys, reconstructed coalitions, identical outputs, and current copied code establish protocol equivalence for ASR. See `derived/blind_protocol_exact_comparison.csv` and `derived/codec_none_exact_comparison.csv`.
"""
    (OUT/"01_blind_protocol_comparison.md").write_text(protocol)

    baseline="""# Classical baseline definitions

Definitions are taken directly from `code_snapshot/scripts/baselines.py`:

- `median`: samplewise median, y[t] = median_i x_i[t].
- `minimum`: samplewise minimum, y[t] = min_i x_i[t].
- `maximum`: samplewise maximum, y[t] = max_i x_i[t].
- `rand_minmax`: independently at every sample, a seeded Bernoulli choice between the samplewise minimum and maximum. It is a conceptual approximation to the randomized min/max literature baseline; this audit does not claim exact equivalence to an external paper formula.
- `copy_paste`: project-defined 20-ms block construction (320 samples at 16 kHz); each output block is copied from one uniformly sampled coalition member.

Thus median/minimum/maximum are exact pointwise numerical operators as coded. Whether a cited paper uses identical preprocessing and tie conventions requires a separate literature audit; it cannot be established from repository code alone.
"""
    (OUT/"02_baseline_definitions.md").write_text(baseline)

    metrics="""# Metric and protocol definitions

1. **ASR**: 1 iff the detector's top-1 registry identity is not a coalition identity; rate is its trial mean.
2. **R3_escape / R5_escape**: 1 iff none of the top 3 / top 5 identities is a colluder.
3. `max_colluder_score` and `max_noncolluder_score`: maxima over coalition and active non-coalition registry identities.
4. Registry/codec/temporal `attribution_margin = max_noncolluder_score - max_colluder_score`; positive favors attack escape.
5. Evidence-chain `margin = max_colluder_score - max_noncolluder_score`; positive favors colluder attribution. It is the opposite sign from `attribution_margin`. `cb_margin` is a legacy column name for the TCT-side colluder margin.
6. `target_margin = score(target) - max score(other active identities)`; positive favors the target.
7. `target_hit` is top-1 identity == target. With exact ties, the script's ordering convention matters; a zero margin does not alone encode the tie winner.
8. Per-target hit uses each target row as denominator. Any-of-ten is 1 per trial iff at least one of its ten targets hits. They are separate files/columns.
9. Favorable target selection samples up to 2,000 deterministic non-colluder candidates, computes exact payload convex-hull distance, and retains the ten smallest. It is nearest within that sampled subset, not necessarily the global registry ten.
10. Arbitrary targets are ten uniformly sampled non-colluder identities without replacement using `coalition_seed + 999`; matched-N samples them from the active N=1024 registry.
11. **FWP = Farthest Waveform Pair**. For all i<j, compute mean_t (x_i[t]-x_j[t])^2, choose the maximizing pair, and output (x_i+x_j)/2. See `scripts/attack.py:fwp` in the snapshot.
12. At K=2 only one pair exists, hence FWP weights are (0.5,0.5), exactly Mean.
13. Native registry is every d-bit identity: 65,536 for 16-bit models and 1,024 for TimbreWM.
14. Matched-N registry uses an independent deterministic SHA256 seed, includes every coalition identity, and samples N-K non-colluders. Detection scores are computed natively and then restricted to the active identities, which is equivalent for ranking within the subset.

**Sign audit:** target margins are consistently positive-for-target; attribution margins are positive-for-escape; evidence-chain margin is deliberately opposite and must not be pooled without a sign conversion.
"""
    (OUT/"03_metric_definitions.md").write_text(metrics)

    voice=qp[qp.model=="voicemark"].groupby(["K","method"])["presence_decision"].agg(["sum","count","mean"]).reset_index().to_markdown(index=False)
    matched_table=pd.DataFrame(source_notes).to_markdown(index=False)
    evcounts=ev_trial.groupby(["model","K"])["rho_wave"].agg(total="size",valid="count").reset_index(); evcounts["nan"]=evcounts.total-evcounts.valid
    report=f"""# Final data audit report

## Executive findings

1. **Blind main result:** use `attack_*` as the paper's native-registry main table; it is the direct 300-trial Mean/FWP experiment and includes quality plus ACC. Use registry-control for registry-size effects. Its native-N ASR exactly reproduces attack, so there is no competing native ASR dataset.
2. **Why native control differs:** it does not differ in ASR. Any earlier aggregate discrepancy is reporting/version selection, not the frozen raw rows. Small WavMark R3/R5 differences come from tied-score sort stability.
3. **FWP:** Farthest Waveform Pair, maximum mean-squared waveform-distance pair mixed 50/50.
4. **Margins:** not globally sign-unified. `attribution_margin` is positive-for-escape; evidence-chain `margin` is positive-for-colluder; `target_margin` is positive-for-target.
5. **WavMark matched-N arbitrary:** **missing as a completed experiment**. K=2 final contains only 20 rows/one smoke trial; K=3/5/8 have no complete final. No partial file is promoted.

## Coverage and row units

- attack: 20×600 = 12,000 method-trial rows, complete.
- registry control: 50,400 model-K-N-method-trial rows, complete.
- evidence chain raw: 20×2,700 = 54,000 method-trial rows (nine methods), not 6,000. The derived Mean/FWP trial view has 6,000 trial rows.
- baselines: 30,000 method-trial rows, complete.
- favorable hull raw: 120,000 method-target rows (Mean and TCT); TCT-only target count is 60,000. Rank is deterministic post-processing by d_hull.
- native arbitrary raw: 120,000 Mean/TCT target rows; the unified TCT detail has 60,000 rows. Non-WavMark files did not save d_hull or target margin, so those fields are NA.
- matched-N arbitrary: 96,000 complete rows for four models, each file containing Mean and TCT; WavMark is incomplete as above.
- quality/presence: 12,000 method-trial rows, complete.
- codec: 9,000 rows = 5 models×3 codecs×2 methods×300 trials, complete.
- temporal: 7,000 rows = 5×7 shifts×2 methods×**100**, not 300, trials. SI_SDR was not saved.
- detector oracle: 160 rows = 2 models×2 K×40 single-target trial contexts.

Matched-N file audit:

{matched_table}

## Alignment missingness

`rho_wave` is undefined at K=2 because one pair cannot support a rank correlation. At K=3 it is undefined when a distance vector is constant/tied: five trials for AudioSeal, VoiceMark, WavMark and WMCodec, and fifteen for TimbreWM. K=5/8 are complete. Therefore pooling K=3/5/8 gives 895 valid trials per former model and 885 for TimbreWM, not 900.

{evcounts.to_markdown(index=False)}

## Quality and presence

All quality values use the first legitimate watermarked coalition copy as reference; none uses clean speech. AudioSeal, VoiceMark and WavMark expose a supported native presence result in the exporter. TimbreWM and WMCodec are marked unsupported by interface design, not failed runs. VoiceMark rates (Mean and FWP kept separate) are:

{voice}

## Sensitivity protocol checks

Codec is applied independently to every coalition copy **before** collusion using ffmpeg MP3/libmp3lame 128 kbps or Opus/libopus 64 kbps; `none` equals attack K=5 ASR on all paired rows. Temporal shifting changes one rotating member only, with fixed-length zero padding and end cropping (not circular shift). Its n=100 schedule contains one trial per speaker (`local_t=0`), rather than the first 100 row positions of the n=300 schedule; after joining by speaker/local_t, the 0-ms condition equals the corresponding attack trials on ASR and stored quality. See `derived/temporal_zero_exact_comparison.csv`.

## What the data can support

Directly supported: native Mean/FWP ASR and quality; matched registry-size effects; model/K classical baselines; favorable versus arbitrary TCT hit rates where the saved endpoint exists; K=5 codec and one-member temporal sensitivity; alignment correlations with explicit valid-N; VoiceMark/WMCodec preliminary oracle comparison.

Limited support: favorable targets are nearest within a 2,000-candidate sample, not globally nearest; native arbitrary geometry/margin is complete only for WavMark; temporal sensitivity has 100 trials/condition and no SI_SDR; detector oracle covers only two models and K=5/8.

Currently unsupported: full WavMark matched-N arbitrary claims; native arbitrary d_hull/margin claims for the other four models; temporal SI_SDR; exact generating commit per individual historical CSV; attack-level coalition/hash, explicit seed and utterance ID because those fields were not stored.

If further runs are authorized, the precise highest-priority gap is WavMark, K=2/3/5/8, N=1024, Mean+TCT, 300 trials×10 targets (6,000 rows per K; replace the K2 one-trial smoke). No experiment was started by this audit. Optional data-only gaps would require rerunning native arbitrary for AudioSeal/TimbreWM/VoiceMark/WMCodec while saving d_hull and target_margin, and temporal all models K=5 Mean/FWP at the desired n=300 while saving SI_SDR.

## Statistical validation and fallacy scan

- Paired Mean/FWP comparisons use identical trial keys; improve/tie/worsen and exact McNemar p-values are exported. Target-row rates are not treated as independent trial rates; any-of-ten is reported separately.
- Multiplicity: aggregate files provide cellwise tests, not multiplicity-adjusted confirmatory claims. Any paper-wide significance claim needs a declared family and correction.
- Confidence intervals: all binary rates include Wilson 95% intervals; continuous summaries report n/mean/SD.
- No linear regression or interpolation across discrete K is performed.
- No subgroup selection, outlier deletion, optional stopping, post-hoc endpoint switching, or causal claim is introduced here. Language splits are descriptive and retain denominators.
- Independence/generalizability: three trials per speaker are repeated measures; rowwise Wilson intervals do not model speaker clustering. Treat them as descriptive unless a speaker-clustered analysis is added.
- Missingness is structural and enumerated, not silently dropped. `rho_wave` valid-N is always shown.
- Practical versus statistical significance must be judged from paired differences and intervals, not p-values alone.
- Measurement validity is limited by native detector score/tie conventions and by use of a watermarked-copy quality reference.

## Material passport

- Experimental objects: five watermark systems, bilingual 100-speaker manifest, deterministic 300-trial protocol.
- Evidence produced: frozen raw CSV copies, harmonized detail/trial files, aggregates, exact protocol joins, code/config snapshot and hashes.
- Assumptions: final non-partial CSVs are authoritative; native detector interfaces and cached embeddings are valid; no unseen historical commit metadata exists in rows.
- Limitations: listed above; especially WavMark matched-N incomplete and repeated-speaker dependence.
- Reproducibility: commit/environment/model identifiers, commands, code SHA256 and every packaged-file SHA256 are included.
"""
    (OUT/"FINAL_AUDIT_REPORT.md").write_text(report)
    readme="""# Paper data audit 2026-08-20

This is a frozen, code-backed audit bundle. It contains no audio, dataset, checkpoint, model weight, credential, or full registry score vector. `raw/` preserves source CSVs unchanged; `derived/` contains deterministic schema harmonization and trial summaries; root CSVs are requested aggregates. Missing original fields remain blank/absent and are documented.

Read in order: `00_environment_and_provenance.md`, `01_blind_protocol_comparison.md`, `02_baseline_definitions.md`, `03_metric_definitions.md`, then `FINAL_AUDIT_REPORT.md`. Validate files with `sha256sum -c SHA256SUMS.txt` from this directory.
"""
    (OUT/"README.md").write_text(readme)


if __name__ == "__main__":
    main()

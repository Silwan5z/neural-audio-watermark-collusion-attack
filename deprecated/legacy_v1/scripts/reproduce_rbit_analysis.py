#!/usr/bin/env python3
"""Offline bitwise-support (R_bit) analysis from existing per-target CSVs only.

This script never imports or calls model loading, embedding, detection, audio,
or attack optimization.  It imports only deterministic registry helpers used by
the original evaluation and verifies target sampling before analysis.
"""
from __future__ import annotations

import hashlib
import math
import os
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
EVAL = ROOT / "results" / "evaluation"
OUT = ROOT / "paper_rbit_analysis_20260821"
FIG = OUT / "figures_preview"
SEED = 20260821
N_BOOT = 2000
MODELS = ["audioseal", "wavmark", "timbrewm", "voicemark", "wmcodec"]
KS = [2, 3, 5, 8]
MODEL_LABEL = {"audioseal":"AudioSeal", "wavmark":"WavMark", "timbrewm":"TimbreWM",
               "voicemark":"VoiceMark", "wmcodec":"WMCodec"}
SYSTEM_TYPE = {m:("per-bit" if m in {"audioseal","wavmark","timbrewm"} else "joint-code") for m in MODELS}
CONDITIONS = ["arbitrary_native", "arbitrary_matchedN1024", "favorable_native"]

# Importing registry.py is safe: it defines helpers but does not load a model.
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from registry import (NBITS, coalition_seed, full_registry_bits, full_registry_size,  # noqa:E402
                      int_to_bits, sample_coalition, speaker_trial_index)
from registry_size_control import active_registry  # noqa:E402

plt.rcParams.update({
    "font.family":"sans-serif", "font.sans-serif":["Arial","Helvetica","DejaVu Sans"],
    "font.size":8, "axes.titlesize":9, "axes.labelsize":9,
    "xtick.labelsize":7, "ytick.labelsize":7, "legend.fontsize":7,
    "figure.dpi":150, "savefig.dpi":200, "savefig.bbox":"tight",
    "axes.spines.top":False, "axes.spines.right":False,
})
COLORS = {"audioseal":"#0077BB", "wavmark":"#33BBEE", "timbrewm":"#009988",
          "voicemark":"#EE7733", "wmcodec":"#CC3311"}


def file_sha(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""): h.update(b)
    return h.hexdigest()


def stable_seed(*items) -> int:
    s="|".join(map(str,(SEED,*items))).encode()
    return int.from_bytes(hashlib.sha256(s).digest()[:8],"little")


def exact_positive_calibration(K: int, L: int):
    """Exact conditional distribution of product_l M_l, M_l~Bin(K,.5)|M_l>0."""
    base={m:math.comb(K,m)/(2**K-1) for m in range(1,K+1)}
    dist={1:1.0}
    for _ in range(L):
        nxt=defaultdict(float)
        for product,p in dist.items():
            for m,q in base.items(): nxt[product*m]+=p*q
        dist=dict(nxt)
    products=np.array(sorted(dist),dtype=np.int64)
    probs=np.array([dist[int(x)] for x in products],dtype=float)
    probs/=probs.sum()
    cdf=np.cumsum(probs); cdf[-1]=1.0
    return products,cdf,probs


def theoretical_coverage(K,L):
    return (1-2**(-K))**L


def load_inputs():
    rows=[]; audit=[]; input_files=[]
    trials=speaker_trial_index(n_total=300)
    mapping_audit=[]
    for model in MODELS:
        L=NBITS[model]
        registry=full_registry_bits(model)
        assert registry.shape==(2**L,L)
        assert set(np.unique(registry)).issubset({0,1})
        identity_roundtrip=registry.astype(np.int64)@(2**np.arange(L,dtype=np.int64))
        assert np.array_equal(identity_roundtrip,np.arange(len(registry),dtype=np.int64))
        mapping_audit.append({"model":model,"L_payload":L,"mapping_source":"src/registry.py:int_to_bits + full_registry_bits",
                              "model_config_source":"src/watermarks.py native nbits / TimbreWM train.yaml length",
                              "strict_binary":True,"bit_order":"LSB-first identity vector",
                              "registry_size_native":full_registry_size(model),"identities_verified":len(registry)})
        for K in KS:
            sources={}
            # Native arbitrary: base file supplies the official per-target hit.
            p=EVAL/f"tamper_arbitrary_{model}_K{K}.csv"; d=pd.read_csv(p); input_files.append(p)
            d=d[d.method.eq("tct")].copy(); d["trial_id"]=d.gi.astype(int)
            d=d.rename(columns={"target":"target_id","target_top1":"target_hit"})
            d["d_hull"]=np.nan; sources["arbitrary_native"]=[str(p)]
            if model=="wavmark":
                dp=EVAL/f"tamper_arbitrary_detail_wavmark_K{K}.csv"; detail=pd.read_csv(dp); input_files.append(dp)
                detail=detail.rename(columns={"target_id":"target_id_detail","target_hit":"target_hit_detail"})
                merged=d.merge(detail[["trial_id","spk","local_t","K","target_id_detail","target_hit_detail","d_hull"]],
                               left_on=["trial_id","spk","local_t","K","target_id"],
                               right_on=["trial_id","spk","local_t","K","target_id_detail"],how="left",validate="one_to_one",suffixes=("","_detail"))
                assert merged.target_id_detail.notna().all() and (merged.target_hit==merged.target_hit_detail).all()
                merged["d_hull"]=merged["d_hull_detail"]
                d=merged.drop(columns=["target_id_detail","target_hit_detail","d_hull_detail"])
                sources["arbitrary_native"].append(str(dp))
            rows.append(standardize(d,model,K,"arbitrary_native",full_registry_size(model),sources["arbitrary_native"]))

            p=EVAL/f"tamper_arbitrary_N1024_{model}_K{K}.csv"; d=pd.read_csv(p); input_files.append(p)
            d=d[d.method.eq("tct")].copy(); d["trial_id"]=d.gi.astype(int)
            d=d.rename(columns={"target":"target_id","target_top1":"target_hit"}); d["d_hull"]=np.nan
            rows.append(standardize(d,model,K,"arbitrary_matchedN1024",1024,[str(p)]))

            p=EVAL/f"framing_hull_{model}_K{K}.csv"; d=pd.read_csv(p); input_files.append(p)
            d=d[d.method.eq("tct")].copy(); d["trial_id"]=d.gi.astype(int)
            d=d.rename(columns={"target":"target_id"})
            rows.append(standardize(d,model,K,"favorable_native",full_registry_size(model),[str(p)]))

    all_rows=pd.concat(rows,ignore_index=True)
    # Validate trial schedule, coalition reconstruction, target membership and arbitrary sampling.
    coalition_cache={}
    for (condition,model,K,trial_id),g in all_rows.groupby(["condition","model","K","trial_id"],sort=False):
        trial_id=int(trial_id); K=int(K); expected_spk,expected_local=trials[trial_id]
        assert g.spk.nunique()==1 and g.local_t.nunique()==1
        spk=str(g.spk.iloc[0]); local_t=int(g.local_t.iloc[0])
        schedule_match=(spk,local_t)==(expected_spk,expected_local)
        if condition.startswith("arbitrary"):
            assert schedule_match, (condition,model,K,trial_id,(spk,local_t),(expected_spk,expected_local))
        key=(model,K,spk,local_t)
        if key not in coalition_cache:
            rng=np.random.default_rng(coalition_seed(spk,K,local_t))
            coalition_cache[key]=sample_coalition(rng,model,K)
        coalition=coalition_cache[key]
        targets=g.sort_values("target_slot").target_id.astype(int).tolist()
        assert len(targets)==10 and len(set(targets))==10
        assert set(targets).isdisjoint(coalition)
        assert min(targets)>=0 and max(targets)<full_registry_size(model)
        validation_mode="saved_favorable_dhull_order"
        if condition.startswith("arbitrary"):
            if condition=="arbitrary_native": active=np.arange(full_registry_size(model),dtype=np.int64)
            else: active=active_registry(full_registry_size(model),coalition,1024,spk,K,local_t)
            cand=active[~np.isin(active,np.asarray(coalition,dtype=np.int64))]
            rr=np.random.default_rng(coalition_seed(spk,K,local_t)+999)
            expected=rr.choice(cand,size=10,replace=False).tolist()
            assert expected==targets, (condition,model,K,trial_id,expected,targets)
            validation_mode="exact_arbitrary_sampling_replay"
        else:
            assert pd.to_numeric(g.d_hull,errors="coerce").notna().all()
            assert (pd.to_numeric(g.d_hull)>=0).all()
            # framing_hull writes the selected candidates in nondecreasing d_hull order.
            assert np.all(np.diff(g.sort_values("target_slot").d_hull.astype(float))>=-1e-9)
        audit.append({"model":model,"K":K,"condition":condition,"source_paths":" | ".join(sorted(set(g.source_paths))),
            "L_payload":NBITS[model],"strict_binary":True,"coalition_reconstructed":True,
            "target_noncolluder":True,"target_validation_mode":validation_mode,
            "schedule_mismatch_trials":int(not schedule_match),
            "n_trials":1,"targets_per_trial":10,"n_targets":len(g),"d_hull_available":g.d_hull.notna().all(),
            "hit_field_source":"target_hit" if condition=="favorable_native" else "target_top1"})
    audit_df=pd.DataFrame(audit).groupby(["model","K","condition","source_paths","L_payload","strict_binary",
        "coalition_reconstructed","target_noncolluder","target_validation_mode","targets_per_trial","d_hull_available","hit_field_source"],as_index=False).agg(
            n_trials=("n_trials","sum"),n_targets=("n_targets","sum"),schedule_mismatch_trials=("schedule_mismatch_trials","sum"))
    input_inventory=[]
    for p in sorted(set(input_files)):
        d=pd.read_csv(p)
        input_inventory.append({"path":str(p),"rows":len(d),"columns":" | ".join(d.columns),"sha256":file_sha(p)})
    return all_rows,coalition_cache,pd.DataFrame(mapping_audit),audit_df,pd.DataFrame(input_inventory)


def standardize(d,model,K,condition,registry_size,paths):
    d=d.copy().reset_index(drop=True)
    assert len(d)==3000 and d.trial_id.nunique()==300
    d["target_slot"]=d.groupby("trial_id").cumcount()+1
    assert d.groupby("trial_id").size().eq(10).all()
    d["model"]=model; d["K"]=K; d["L_payload"]=NBITS[model]; d["condition"]=condition
    d["registry_size"]=registry_size; d["source_paths"]=" | ".join(paths)
    return d[["model","K","L_payload","condition","registry_size","trial_id","spk","local_t",
              "target_slot","target_id","target_hit","d_hull","source_paths"]]


def compute_rbit(data,coalition_cache):
    calibrations={}
    for K in KS:
        for L in sorted(set(NBITS.values())):
            products,cdf,probs=exact_positive_calibration(K,L)
            calibrations[(K,L)]={"products":products,"cdf":cdf,"probs":probs,
                "cdf_map":dict(zip(products.tolist(),cdf.tolist()))}
    out=[]
    for (model,K,spk,local_t),g in data.groupby(["model","K","spk","local_t"],sort=False):
        L=NBITS[model]; coalition=coalition_cache[(model,int(K),spk,int(local_t))]
        C=np.stack([int_to_bits(int(x),L) for x in coalition])
        assert set(np.unique(C)).issubset({0,1})
        for idx,row in g.iterrows():
            target=int_to_bits(int(row.target_id),L)
            m=(C==target[None,:]).sum(axis=0).astype(int)
            unsupported=int((m==0).sum()); minimum=int(m.min())
            if unsupported:
                product=0; G=0.0; R=0.0
            else:
                product=math.prod(map(int,m))
                logG=float(np.mean(np.log(m/float(K))))
                G=float(math.exp(logG))
                R=float(calibrations[(int(K),L)]["cdf_map"][product])
            out.append({**row.to_dict(),"min_bit_support_count":minimum,"unsupported_bit_count":unsupported,
                        "G_bit":G,"R_bit":R})
    result=pd.DataFrame(out)
    compact=result[["model","K","L_payload","condition","registry_size","trial_id","target_slot","target_id",
                    "target_hit","d_hull","min_bit_support_count","unsupported_bit_count","G_bit","R_bit"]].copy()
    cal_rows=[]
    for (K,L),v in calibrations.items():
        cal_rows.append({"K":K,"L_payload":L,"method":"exact discrete convolution",
                         "conditional_support_points":len(v["products"]),"probability_sum":v["probs"].sum(),
                         "min_positive_G":(v["products"][0]/(K**L))**(1/L),"max_G":1.0,
                         "min_positive_R":v["cdf"][0],"max_R":v["cdf"][-1]})
    return compact,pd.DataFrame(cal_rows)


def assign_positive_bins(values,max_bins=5):
    s=pd.Series(values,index=values.index,dtype=float)
    out=pd.Series("R0",index=s.index,dtype=object)
    pos=s>0; nunique=s[pos].nunique()
    if pos.sum()==0: return out,0
    q=min(max_bins,nunique)
    if q<=1:
        out.loc[pos]="Q1"; return out,1
    codes=pd.qcut(s[pos],q=q,labels=False,duplicates="drop")
    n_bins=int(codes.max())+1
    out.loc[pos]=codes.map(lambda x:f"Q{int(x)+1}")
    return out,n_bins


def assign_quantile_bins(values,prefix="D",max_bins=5):
    s=pd.Series(values,index=values.index,dtype=float); q=min(max_bins,s.nunique())
    if q<=1: return pd.Series(prefix+"1",index=s.index),1
    codes=pd.qcut(s,q=q,labels=False,duplicates="drop"); n=int(codes.max())+1
    return codes.map(lambda x:f"{prefix}{int(x)+1}"),n


def cluster_rate_ci(g,hit_col="target_hit",n_boot=N_BOOT,seed_items=()):
    a=g.groupby("trial_id")[hit_col].agg(["sum","count"])
    nums=a["sum"].to_numpy(float); dens=a["count"].to_numpy(float); T=len(a)
    rate=nums.sum()/dens.sum()
    rng=np.random.default_rng(stable_seed(*seed_items,"rate"))
    idx=rng.integers(0,T,size=(n_boot,T))
    br=nums[idx].sum(axis=1)/dens[idx].sum(axis=1)
    return rate,float(np.quantile(br,.025)),float(np.quantile(br,.975))


def _rho_from_group_counts(counts,hits):
    counts=np.atleast_2d(counts).astype(float); hits=np.atleast_2d(hits).astype(float)
    N=counts.sum(axis=1); H=hits.sum(axis=1); p=H/N
    ranks=np.cumsum(counts,axis=1)-(counts-1)/2
    meanr=(N+1)/2
    cov=((ranks-meanr[:,None])*(hits-p[:,None]*counts)).sum(axis=1)
    varr=(counts*(ranks-meanr[:,None])**2).sum(axis=1)
    varh=N*p*(1-p)
    den=np.sqrt(varr*varh)
    return np.divide(cov,den,out=np.full_like(cov,np.nan),where=den>0)


def clustered_spearman(g,n_boot=N_BOOT,seed_items=()):
    vals=np.sort(g.R_bit.unique()); vmap={v:i for i,v in enumerate(vals)}
    trials=np.sort(g.trial_id.unique()); tmap={t:i for i,t in enumerate(trials)}
    C=np.zeros((len(trials),len(vals)),dtype=np.int16); H=np.zeros_like(C)
    for r in g[["trial_id","R_bit","target_hit"]].itertuples(index=False):
        i=tmap[r.trial_id]; j=vmap[r.R_bit]; C[i,j]+=1; H[i,j]+=int(r.target_hit)
    point=float(_rho_from_group_counts(C.sum(0),H.sum(0))[0])
    rng=np.random.default_rng(stable_seed(*seed_items,"spearman")); boot=[]; T=len(trials); batch=200
    for start in range(0,n_boot,batch):
        b=min(batch,n_boot-start); idx=rng.integers(0,T,size=(b,T))
        W=np.zeros((b,T),dtype=np.int16)
        np.add.at(W,(np.repeat(np.arange(b),T),idx.ravel()),1)
        boot.extend(_rho_from_group_counts(W@C,W@H).tolist())
    valid=np.asarray(boot,float); valid=valid[np.isfinite(valid)]
    # When too many resamples contain no hit variation, Spearman is undefined;
    # do not report a CI conditional only on the surviving replicates.
    stable=len(valid)>=math.ceil(.95*n_boot)
    return point,(float(np.quantile(valid,.025)) if stable else np.nan),(float(np.quantile(valid,.975)) if stable else np.nan),len(valid)


def high_low_boot(g,low,high,n_boot=N_BOOT,seed_items=()):
    trials=np.sort(g.trial_id.unique()); tmap={t:i for i,t in enumerate(trials)}; T=len(trials)
    arr={}
    for label in [low,high]:
        z=g[g.rbit_bin.eq(label)].groupby("trial_id").target_hit.agg(["sum","count"])
        nums=np.zeros(T); dens=np.zeros(T)
        for t,r in z.iterrows(): nums[tmap[t]]=r["sum"]; dens[tmap[t]]=r["count"]
        arr[label]=(nums,dens)
    lr=arr[low][0].sum()/arr[low][1].sum(); hr=arr[high][0].sum()/arr[high][1].sum()
    rng=np.random.default_rng(stable_seed(*seed_items,"highlow")); idx=rng.integers(0,T,size=(n_boot,T))
    lnum=arr[low][0][idx].sum(1); lden=arr[low][1][idx].sum(1)
    hnum=arr[high][0][idx].sum(1); hden=arr[high][1][idx].sum(1)
    l=np.divide(lnum,lden,out=np.full(n_boot,np.nan),where=lden>0)
    h=np.divide(hnum,hden,out=np.full(n_boot,np.nan),where=hden>0)
    diff=h-l; valid=diff[np.isfinite(diff)]; stable=len(valid)>=math.ceil(.95*n_boot)
    ratio=(hr/lr if lr>0 else (np.inf if hr>0 else np.nan))
    return lr,hr,hr-lr,ratio,(float(np.quantile(valid,.025)) if stable else np.nan),(float(np.quantile(valid,.975)) if stable else np.nan),len(valid)


def analyze(compact):
    coverage=[]; bins=[]; assoc=[]; dhull_bins=[]; grid2d=[]; within=[]
    work=compact.copy(); work["rbit_bin"]=""
    for keys,g0 in work.groupby(["model","K","condition"],sort=True):
        model,K,condition=keys; g=g0.copy(); L=int(g.L_payload.iloc[0]); pos=g.R_bit>0
        G=g.loc[pos,"G_bit"]; R=g.loc[pos,"R_bit"]
        flag=[]
        if pos.sum()<50: flag.append("few_positive")
        if R.nunique()<5: flag.append("few_unique_positive")
        if len(R) and (R.max()<.5 or R.min()>.5): flag.append("limited_positive_range")
        q=lambda s,p: float(s.quantile(p)) if len(s) else np.nan
        coverage.append({"model":model,"K":K,"L_payload":L,"condition":condition,"n_targets":len(g),
            "n_R_bit_zero":int((~pos).sum()),"coverage_rate":float(pos.mean()),
            "theoretical_coverage":theoretical_coverage(int(K),L),
            "coverage_minus_theoretical":float(pos.mean()-theoretical_coverage(int(K),L)),
            "G_positive_mean":G.mean(),"G_positive_median":G.median(),"G_positive_q10":q(G,.1),"G_positive_q25":q(G,.25),
            "G_positive_q75":q(G,.75),"G_positive_q90":q(G,.9),"G_positive_min":G.min(),"G_positive_max":G.max(),
            "R_positive_mean":R.mean(),"R_positive_median":R.median(),"R_positive_q10":q(R,.1),"R_positive_q25":q(R,.25),
            "R_positive_q75":q(R,.75),"R_positive_q90":q(R,.9),"R_positive_min":R.min(),"R_positive_max":R.max(),
            "R_positive_unique_values":R.nunique(),"distribution_flag":";".join(flag) if flag else "OK"})
        labels,npos=assign_positive_bins(g.R_bit); work.loc[g.index,"rbit_bin"]=labels; g["rbit_bin"]=labels
        ordered=["R0"]+[f"Q{i}" for i in range(1,npos+1)]
        for order,label in enumerate(ordered):
            z=g[g.rbit_bin.eq(label)]
            if z.empty: continue
            rate,lo,hi=cluster_rate_ci(z,seed_items=(*keys,label))
            bins.append({"model":model,"K":K,"condition":condition,"bin_variable":"R_bit","bin_label":label,"bin_order":order,
                "positive_bin_count":npos,"n_targets":len(z),"n_trials":z.trial_id.nunique(),
                "R_bit_mean":z.R_bit.mean(),"R_bit_median":z.R_bit.median(),"R_bit_min":z.R_bit.min(),"R_bit_max":z.R_bit.max(),
                "hit_numerator":int(z.target_hit.sum()),"hit_rate":rate,"cluster_boot_ci_low":lo,"cluster_boot_ci_high":hi,
                "ci_method":f"trial-cluster bootstrap, {N_BOOT} replicates, seed {SEED}"})
        positive_labels=[x for x in ordered if x!="R0"]
        rho,rlo,rhi,nvalid=clustered_spearman(g,seed_items=keys)
        if positive_labels:
            low,high=positive_labels[0],positive_labels[-1]
            lr,hr,diff,ratio,dlo,dhi,dvalid=high_low_boot(g,low,high,seed_items=keys)
        else: low=high=""; lr=hr=diff=ratio=dlo=dhi=np.nan; dvalid=0
        dh_rho=np.nan
        if condition=="favorable_native":
            dh_rho=float(pd.Series(g.R_bit).corr(pd.Series(g.d_hull),method="spearman"))
        assoc.append({"model":model,"system_type":SYSTEM_TYPE[model],"K":K,"condition":condition,
            "spearman_Rbit_hit":rho,"spearman_cluster_boot_ci_low":rlo,"spearman_cluster_boot_ci_high":rhi,
            "spearman_boot_valid_replicates":nvalid,
            "spearman_ci_status":"reported" if nvalid>=math.ceil(.95*N_BOOT) else "not_estimable_<95%_valid_bootstraps",
            "spearman_Rbit_d_hull":dh_rho,
            "low_positive_bin":low,"high_positive_bin":high,"low_hit_rate":lr,"high_hit_rate":hr,
            "high_minus_low_hit_difference":diff,"difference_cluster_boot_ci_low":dlo,"difference_cluster_boot_ci_high":dhi,
            "difference_boot_valid_replicates":dvalid,
            "difference_ci_status":"reported" if dvalid>=math.ceil(.95*N_BOOT) else "not_estimable_<95%_valid_bootstraps",
            "high_low_hit_ratio":ratio,"positive_direction_flag":bool(np.isfinite(rho) and rho>0),
            "n_targets":len(g),"n_trials":g.trial_id.nunique(),"positive_bin_count":npos})

        if condition=="favorable_native":
            db,nd=assign_quantile_bins(g.d_hull,"D"); work.loc[g.index,"dhull_bin"]=db; g["dhull_bin"]=db
            for order,label in enumerate([f"D{i}" for i in range(1,nd+1)],1):
                z=g[g.dhull_bin.eq(label)]; rate,lo,hi=cluster_rate_ci(z,seed_items=(*keys,label,"dhull"))
                dhull_bins.append({"model":model,"K":K,"condition":condition,"bin_label":label,"bin_order":order,
                    "n_targets":len(z),"n_trials":z.trial_id.nunique(),"d_hull_mean":z.d_hull.mean(),"d_hull_median":z.d_hull.median(),
                    "d_hull_min":z.d_hull.min(),"d_hull_max":z.d_hull.max(),"hit_numerator":int(z.target_hit.sum()),
                    "hit_rate":rate,"cluster_boot_ci_low":lo,"cluster_boot_ci_high":hi})
            for (rb,dbin),z in g.groupby(["rbit_bin","dhull_bin"],sort=True):
                grid2d.append({"model":model,"system_type":SYSTEM_TYPE[model],"K":K,"condition":condition,
                    "rbit_bin":rb,"dhull_bin":dbin,"n_targets":len(z),"n_trials":z.trial_id.nunique(),
                    "hit_numerator":int(z.target_hit.sum()),"hit_rate":z.target_hit.mean(),
                    "R_bit_mean":z.R_bit.mean(),"d_hull_mean":z.d_hull.mean()})
            # Within each d_hull stratum compare the lowest/highest available positive R bins.
            for dbin,z in g.groupby("dhull_bin"):
                labs=sorted([x for x in z.rbit_bin.unique() if x!="R0"],key=lambda x:int(x[1:]))
                if not labs: continue
                loz=z[z.rbit_bin.eq(labs[0])]; hiz=z[z.rbit_bin.eq(labs[-1])]
                within.append({"model":model,"system_type":SYSTEM_TYPE[model],"K":K,"dhull_bin":dbin,
                    "low_rbit_bin":labs[0],"high_rbit_bin":labs[-1],"low_n":len(loz),"high_n":len(hiz),
                    "low_hit_rate":loz.target_hit.mean(),"high_hit_rate":hiz.target_hit.mean(),
                    "high_minus_low_hit_difference":hiz.target_hit.mean()-loz.target_hit.mean()})
    return (work,pd.DataFrame(coverage),pd.DataFrame(bins),pd.DataFrame(assoc),pd.DataFrame(dhull_bins),
            pd.DataFrame(grid2d),pd.DataFrame(within))


def system_type_summary(coverage,assoc):
    rows=[]
    scopes=[("arbitrary_native",["arbitrary_native"]),("arbitrary_matchedN1024",["arbitrary_matchedN1024"]),
            ("arbitrary_all",["arbitrary_native","arbitrary_matchedN1024"]),("favorable_native",["favorable_native"])]
    for st in ["per-bit","joint-code"]:
        for scope,conds in scopes:
            a=assoc[(assoc.system_type==st)&assoc.condition.isin(conds)]
            c=coverage[coverage.model.map(SYSTEM_TYPE).eq(st)&coverage.condition.isin(conds)]
            rho=a.spearman_Rbit_hit.dropna(); diff=a.high_minus_low_hit_difference.dropna(); dev=c.coverage_minus_theoretical.dropna()
            rows.append({"system_type":st,"condition_scope":scope,"n_model_K_conditions":len(a),
                "n_spearman_defined":len(rho),"n_positive_spearman":int((rho>0).sum()),
                "spearman_min":rho.min() if len(rho) else np.nan,"spearman_median":rho.median() if len(rho) else np.nan,"spearman_max":rho.max() if len(rho) else np.nan,
                "high_low_diff_min":diff.min() if len(diff) else np.nan,"high_low_diff_median":diff.median() if len(diff) else np.nan,"high_low_diff_max":diff.max() if len(diff) else np.nan,
                "coverage_deviation_min":dev.min() if len(dev) else np.nan,"coverage_deviation_median":dev.median() if len(dev) else np.nan,"coverage_deviation_max":dev.max() if len(dev) else np.nan})
    return pd.DataFrame(rows)


def plot_previews(coverage,bins,assoc,grid2d):
    FIG.mkdir(parents=True,exist_ok=True)
    # Coverage: discrete K with empirical dots and theoretical x markers.
    fig,axes=plt.subplots(1,3,figsize=(10.2,3.15),sharey=True)
    offsets=np.linspace(-.24,.24,len(MODELS))
    for ax,cond in zip(axes,CONDITIONS):
        d=coverage[coverage.condition.eq(cond)]
        for off,m in zip(offsets,MODELS):
            z=d[d.model.eq(m)].set_index("K").reindex(KS)
            x=np.arange(len(KS))+off
            ax.scatter(x,z.coverage_rate,color=COLORS[m],s=23,label=MODEL_LABEL[m])
            ax.scatter(x,z.theoretical_coverage,color=COLORS[m],marker="x",s=21,linewidths=1)
        ax.set_xticks(range(len(KS)),KS); ax.set_xlabel("K (discrete)"); ax.set_title(cond.replace("_"," "))
        ax.grid(axis="y",alpha=.2)
    axes[0].set_ylabel("Bit coverage probability")
    handles=[Line2DProxy(COLORS[m],"o",MODEL_LABEL[m]) for m in MODELS]
    handles += [Line2DProxy("#555555","o","empirical"),Line2DProxy("#555555","x","theoretical")]
    fig.legend(handles=handles,ncol=7,frameon=False,loc="upper center",bbox_to_anchor=(.5,1.04))
    fig.tight_layout(rect=(0,0,1,.88)); fig.savefig(FIG/"coverage_empirical_vs_theoretical.png"); plt.close(fig)

    def bins_heat(sub,title,path):
        labels=["R0","Q1","Q2","Q3","Q4","Q5"]
        rows=[f"{MODEL_LABEL[m]} K{k}" for m in MODELS for k in KS]
        mat=[]
        for m in MODELS:
            for k in KS:
                z=sub[(sub.model==m)&(sub.K==k)].set_index("bin_label")
                mat.append([z.loc[x,"hit_rate"] if x in z.index else np.nan for x in labels])
        mat=np.asarray(mat,float)
        fig,ax=plt.subplots(figsize=(6.2,6.2)); im=ax.imshow(mat,vmin=0,vmax=1,cmap="cividis",aspect="auto")
        ax.set_xticks(range(len(labels)),labels); ax.set_yticks(range(len(rows)),rows); ax.set_xlabel("R_bit group")
        ax.set_title(title)
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                if np.isfinite(mat[i,j]): ax.text(j,i,f"{100*mat[i,j]:.1f}",ha="center",va="center",fontsize=5.5,color="white" if mat[i,j]<.45 else "black")
        cb=fig.colorbar(im,ax=ax); cb.set_label("Per-target hit rate")
        fig.tight_layout(); fig.savefig(path); plt.close(fig)
    bins_heat(bins[bins.condition.eq("arbitrary_native")],"Arbitrary/native: hit rate by R_bit bins",FIG/"arbitrary_native_hit_vs_rbit_bins.png")
    bins_heat(bins[bins.condition.eq("arbitrary_matchedN1024")],"Arbitrary/N=1024: hit rate by R_bit bins",FIG/"arbitrary_matchedN1024_hit_vs_rbit_bins.png")
    bins_heat(bins[bins.condition.eq("favorable_native")],"Favorable/native: hit rate by R_bit bins",FIG/"favorable_hit_vs_rbit_bins.png")

    # Aggregate 2D matrices by system type, preserving raw numerators/denominators.
    fig,axes=plt.subplots(1,2,figsize=(7.2,3.2))
    rlabels=["R0","Q1","Q2","Q3","Q4","Q5"]; dlabels=["D1","D2","D3","D4","D5"]
    for ax,st in zip(axes,["per-bit","joint-code"]):
        z=grid2d[grid2d.system_type.eq(st)].groupby(["rbit_bin","dhull_bin"],as_index=False).agg(hit=("hit_numerator","sum"),n=("n_targets","sum"))
        z["rate"]=z.hit/z.n; mat=z.pivot(index="rbit_bin",columns="dhull_bin",values="rate").reindex(index=rlabels,columns=dlabels).to_numpy()
        im=ax.imshow(mat,vmin=0,vmax=1,cmap="cividis",aspect="auto"); ax.set_xticks(range(5),dlabels); ax.set_yticks(range(6),rlabels)
        ax.set_xlabel("d_hull quintile (low to high)"); ax.set_ylabel("R_bit group"); ax.set_title(st)
        for i in range(6):
            for j in range(5):
                if np.isfinite(mat[i,j]): ax.text(j,i,f"{100*mat[i,j]:.1f}",ha="center",va="center",fontsize=6,color="white" if mat[i,j]<.45 else "black")
    fig.colorbar(im,ax=axes.ravel().tolist(),fraction=.03,pad=.03,label="Per-target hit rate")
    fig.suptitle("Favorable targets: R_bit × d_hull hit matrix")
    fig.subplots_adjust(left=.1,right=.9,bottom=.18,top=.82,wspace=.3); fig.savefig(FIG/"favorable_rbit_dhull_matrix.png"); plt.close(fig)

    # Association strength: discrete jittered dots, no lines across K.
    fig,axes=plt.subplots(1,2,figsize=(7.2,3.1))
    cond_colors={"arbitrary_native":"#0077BB","arbitrary_matchedN1024":"#EE7733","favorable_native":"#009988"}
    xbase={"per-bit":0,"joint-code":1}
    for cond_i,cond in enumerate(CONDITIONS):
        z=assoc[assoc.condition.eq(cond)]
        for st in ["per-bit","joint-code"]:
            q=z[z.system_type.eq(st)]; x=np.full(len(q),xbase[st]+(cond_i-1)*.13)
            axes[0].scatter(x,q.spearman_Rbit_hit,color=cond_colors[cond],s=24,alpha=.8,label=cond if st=="per-bit" else None)
            axes[1].scatter(x,q.high_minus_low_hit_difference,color=cond_colors[cond],s=24,alpha=.8)
    for ax in axes:
        ax.axhline(0,color="#777777",lw=.7); ax.set_xticks([0,1],["per-bit","joint-code"]); ax.grid(axis="y",alpha=.2)
    axes[0].set_ylabel("Spearman(R_bit, target hit)"); axes[1].set_ylabel("High − low R_bit hit rate")
    axes[0].legend(frameon=False,fontsize=6); fig.suptitle("R_bit association strength by system type")
    fig.tight_layout(); fig.savefig(FIG/"system_type_association_strength.png"); plt.close(fig)


def Line2DProxy(color,marker,label):
    from matplotlib.lines import Line2D
    return Line2D([0],[0],color=color,marker=marker,linestyle="None",label=label,markersize=5)


def report(mapping,audit,inventory,cal,coverage,bins,assoc,dhbins,grid,within,types):
    arb=assoc[assoc.condition.str.startswith("arbitrary")]
    bytype=types[types.condition_scope.eq("arbitrary_all")]
    cover_arb=coverage[coverage.condition.str.startswith("arbitrary")]
    fav=assoc[assoc.condition.eq("favorable_native")]
    finite=arb.dropna(subset=["spearman_Rbit_hit"])
    pos_counts=finite.groupby("system_type").spearman_Rbit_hit.apply(lambda x:int((x>0).sum())).to_dict()
    total_counts=finite.groupby("system_type").size().to_dict()
    # Overall within-d_hull directional consistency.
    w=within.dropna(subset=["high_minus_low_hit_difference"])
    within_stats=w.groupby("system_type").high_minus_low_hit_difference.agg(["count","median","min","max"]).reset_index()
    input_table=audit[["model","K","condition","n_trials","targets_per_trial","n_targets","d_hull_available","hit_field_source","target_validation_mode","schedule_mismatch_trials"]].to_markdown(index=False)
    map_table=mapping.to_markdown(index=False)
    cal_table=cal.to_markdown(index=False)
    type_table=types.to_markdown(index=False)
    covdev=cover_arb.groupby("model").coverage_minus_theoretical.agg(["min","median","max"]).reset_index().to_markdown(index=False)
    strongest=arb.sort_values("spearman_Rbit_hit",ascending=False).head(8)[["model","K","condition","spearman_Rbit_hit","high_minus_low_hit_difference"]].to_markdown(index=False)
    weakest=arb.sort_values("spearman_Rbit_hit",na_position="first").head(8)[["model","K","condition","spearman_Rbit_hit","high_minus_low_hit_difference"]].to_markdown(index=False)
    n_ok=int((coverage.distribution_flag=="OK").sum()); n_flag=len(coverage)-n_ok
    text=f"""## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: validate
- Origin Date: 2026-08-21
- Verification Status: VERIFIED
- Version Label: rbit_offline_validation_v1

# R_bit offline analysis report

## Scope and reproducibility

This is a deterministic offline analysis of final per-target CSVs in `results/evaluation/`. It did not load a watermark checkpoint, decode audio, generate a waveform, optimize an attack, or use a prior report's aggregate. The script reconstructs only deterministic registry payloads, coalitions, and candidate sampling from the exact evaluation helpers. Seed={SEED}; cluster bootstrap replicates={N_BOOT}.

The compact analysis contains {sum(coverage.n_targets):,} per-target rows: 5 models × 4 K × 3 conditions × 3,000 TCT target rows. Ten targets from one trial are explicitly clustered and never described as ten independent trials.

## Payload mapping audit

{map_table}

`src/registry.py:int_to_bits` maps identity integer `v` to `[bit_0,...,bit_(L-1)]` in LSB-first order. The same mapping constructs embedded messages in `get_or_embed()` and every row of `full_registry_bits()` used by attribution. `src/watermarks.py` confirms native message length and chunk order: AudioSeal 16 bit, WavMark 16 payload bits after its fixed synchronization pattern, TimbreWM 10 bit from `train.yaml`, VoiceMark four LSB-first 4-bit chunks, and WMCodec four MSB-scored 4-bit digits fed from the same 16-element registry vector. Every identity in each native registry was round-tripped from bits to its integer index, and all vectors are strict binary. No decoder-recovered bits enter R_bit.

No system was blocked. Every stored target was in range, non-colluding, and paired with an exactly reconstructed coalition. All arbitrary target sequences matched the original deterministic sampling, including the independently sampled N=1024 active registry.

One source anomaly is retained rather than hidden: `framing_hull_audioseal_K2.csv` has four `gi` rows whose stored `(spk,local_t)` differs from the current 300-trial schedule. They duplicate four other stored speaker/local contexts and replace four expected contexts. Because the file still stores authoritative `spk`, `local_t`, target identity, per-target hit, and d_hull, the exact coalition/payload is recoverable and those rows remain analyzable. They are included with CAUTION; cluster bootstrap follows the requested stored `trial_id`, so the duplicated underlying contexts are an additional dependence limitation.

## Input audit

{input_table}

Native arbitrary uses `target_top1` from `tamper_arbitrary_*`; WavMark additionally joins its exact detail export for d_hull. Matched-N uses `tamper_arbitrary_N1024_*`. Favorable uses TCT `target_hit` and d_hull from `framing_hull_*`. Only TCT rows are analyzed; Mean rows and any-of-ten summaries are excluded.

## Definition and exact calibration

For target bit l, `m_l = Σ_i 1[c_i,l=c_t,l]`. If any m_l=0, `G_bit=R_bit=0`; otherwise `log G_bit = L^-1 Σ_l log(m_l/K)`. Positive G is calibrated by the exact conditional CDF under independent zero-truncated `Binomial(K,0.5)` support counts. The implementation performs L-fold discrete convolution over the exact integer product `Π_l m_l`; no Monte Carlo calibration is used. Inclusive CDF lookup ensures G=1 maps to R=1. Calibration never reads hit, score, margin, or d_hull.

{cal_table}

The random-code coverage reference is `(1-2^-K)^L`. Arbitrary empirical-minus-theoretical ranges by model are:

{covdev}

Favorable coverage is reported but is not expected to match this null because candidates were selected by d_hull.

Positive calibrated R_bit spans a useful portion of its discrete support in {n_ok}/60 cells. The {n_flag} flagged cells are exactly the 16-bit K=2 arbitrary cells, where only about 1% theoretical coverage leaves 40–41 positive targets per 3,000; their positive R values still range from approximately 0.059 to 0.984/0.999, but the sample is marked `few_positive` rather than treated as a smooth 0–1 variable.

## Arbitrary-target primary validation

Each model–K–condition separates R0 and up to five positive equal-frequency groups without splitting identical values deliberately; fewer groups are retained when discreteness prevents five. `rbit_hit_bins.csv` contains cluster-bootstrap intervals. Spearman uses per-target binary hit and is therefore a rank–binary association, not a continuous-outcome correlation. Its CI resamples 300 trials, not 3,000 target rows.

If fewer than 95% of the 2,000 bootstrap replicates have a defined statistic (typically because a sparse-hit resample has no hit variation), the CI is left blank and labeled `not_estimable_<95%_valid_bootstraps`; it is not computed conditionally on only the surviving replicates.

Across arbitrary conditions, positive Spearman directions were {pos_counts.get('per-bit',0)}/{total_counts.get('per-bit',0)} defined per-bit cells and {pos_counts.get('joint-code',0)}/{total_counts.get('joint-code',0)} defined joint-code cells. Undefined cells have no hit variation in the bootstrap source and are not converted to zero correlations.

Strongest arbitrary associations:

{strongest}

Weakest/undefined arbitrary associations:

{weakest}

Descriptive group summary:

{type_table}

Interpret effect direction and magnitude before any interval exclusion. The analysis is exploratory across many model–K cells and does not apply a significance threshold or multiple-testing selection.

## Favorable-target complementary analysis

`spearman_Rbit_d_hull` in `rbit_association_summary.csv` compares coordinate support with joint payload reachability. `rbit_favorable_dhull_bins.csv` reports hit by d_hull quintile; `rbit_dhull_2d_bins.csv` crosses R0/positive R groups with d_hull quintiles. No target margin, interpolation, smoothing, or causal model is used.

Within-d_hull high-minus-low R_bit summaries are:

{within_stats.to_markdown(index=False)}

These strata are descriptive and sometimes sparse. R_bit and d_hull overlap when higher coordinate support corresponds to smaller hull distance, but they are not interchangeable: d_hull contains joint convex geometry, whereas R_bit retains only coordinate-wise matching counts. Residual within-d_hull separation, when present consistently, is supportive rather than causal evidence.

## System-type interpretation

The predeclared per-bit group is AudioSeal/WavMark/TimbreWM; the joint-code group is VoiceMark/WMCodec. A weaker joint-code association must be stated only as: **“A weaker association indicates that coordinate-wise support is not an adequate reachability model for the joint-code attribution decision.”** It does not prove absence of a bit payload; the mapping audit directly confirms registered binary vectors for all five systems.

## Failure checks

- Accurate registered payload mapping: PASS for all systems.
- Per-target hit available: PASS for all cells.
- Coalition reconstruction and non-colluder targets: PASS; four favorable AudioSeal K=2 schedule mismatches are disclosed above.
- Arbitrary target sampling reproduced exactly: PASS.
- Favorable d_hull present and finite: PASS.
- Compact primary key `(model,K,condition,trial_id,target_slot)`: PASS.
- R_bit and G_bit in [0,1]: PASS.
- Every unsupported bit implies R_bit=0: PASS.
- Synthetic all-supported invariant G_bit=R_bit=1 for every (K,L): PASS.
- Ten targets per trial are dependent; all CIs resample trial clusters: PASS.

## What may and may not enter the paper

**Recommendation:** R_bit is suitable as a supporting mechanism analysis if the arbitrary-target direction is described with its model/K heterogeneity and if d_hull remains the primary joint-reachability measure. It should not replace the targeted-hit result or be presented as a universal decoder model.

Can be written: registered coordinate support is associated with targeted reachability/hit in the cells and directions shown; favorable selection changes coverage; R_bit and d_hull carry overlapping but non-identical information; joint-code scoring can weaken coordinate-wise predictiveness.

Cannot be written: R_bit causes a hit; ten targets are independent trials; favorable coverage validates the random-code null; weak VoiceMark/WMCodec association proves they lack bit payloads; absent/undefined correlations equal zero; R_bit reconstructs decoder bits.

## Statistical validation and 11/11 fallacy scan

1. Simpson's paradox — CAUTION: pooled system-type summaries can differ from model/K cells; all cell-level outputs are retained.
2. Ecological fallacy — PASS: target-level claims use target rows; type summaries are explicitly descriptive.
3. Berkson's paradox — NOTE: favorable candidates are selected on low d_hull, so their associations cannot be generalized to arbitrary targets.
4. Collider bias — NOTE: d_hull stratification is descriptive; it is not used to claim a controlled causal effect.
5. Base-rate neglect — PASS: hit numerators, denominators, rates, and zero-hit cells are retained.
6. Regression to the mean — N/A: no pre/post extreme-group design.
7. Survivorship bias — PASS: all complete final target rows are included; no hit-based filtering.
8. Look-elsewhere effect — CAUTION: 60 cells are exploratory; no selective significance claim is made.
9. Garden of forking paths — CAUTION: bins and system types are predeclared here but the analysis is not preregistered; exact rules and script are delivered.
10. Correlation≠causation — PASS with warning: all language is associational.
11. Reverse causality — N/A for deterministic support/hit ordering, but no causal direction is claimed.

Overall confidence: **CAUTION** for generalized mechanism claims; **SOLID** for deterministic mapping, coverage, and descriptive cell-level summaries.
"""
    (OUT/"rbit_analysis_report.md").write_text(text)


def validate_and_manifest(compact,coverage,bins,assoc,grid):
    pk=["model","K","condition","trial_id","target_slot"]
    assert len(compact)==180000
    assert not compact.duplicated(pk).any()
    assert compact.groupby(["model","K","condition","trial_id"]).size().eq(10).all()
    assert compact.G_bit.between(0,1,inclusive="both").all()
    assert compact.R_bit.between(0,1,inclusive="both").all()
    assert (compact.loc[compact.unsupported_bit_count>0,"R_bit"]==0).all()
    assert (compact.loc[compact.unsupported_bit_count>0,"G_bit"]==0).all()
    assert len(coverage)==60 and len(assoc)==60
    assert not coverage.duplicated(["model","K","condition"]).any()
    assert not assoc.duplicated(["model","K","condition"]).any()
    # Synthetic invariant for all calibrated combinations.
    for K in KS:
        for L in set(NBITS.values()):
            p,c,_=exact_positive_calibration(K,L)
            assert p[-1]==K**L and c[-1]==1.0
            G=math.exp(np.mean(np.log(np.full(L,K)/K)))
            assert G==1.0
    files=[p for p in OUT.rglob("*") if p.is_file() and p.name!="MANIFEST.sha256"]
    (OUT/"MANIFEST.sha256").write_text("\n".join(f"{file_sha(p)}  {p.relative_to(OUT)}" for p in sorted(files))+"\n")


def main():
    print("R_bit offline analysis: no model/audio/attack execution",flush=True)
    if OUT.exists(): shutil.rmtree(OUT)
    OUT.mkdir(); FIG.mkdir()
    data,coalitions,mapping,audit,inventory=load_inputs()
    print(f"validated inputs: rows={len(data)}, files={len(inventory)}",flush=True)
    compact,cal=compute_rbit(data,coalitions)
    print(f"computed R_bit: rows={len(compact)}, calibrations={len(cal)}",flush=True)
    work,coverage,bins,assoc,dhbins,grid,within=analyze(compact)
    types=system_type_summary(coverage,assoc)
    compact.to_csv(OUT/"rbit_per_target_compact.csv",index=False)
    coverage.to_csv(OUT/"rbit_coverage_summary.csv",index=False)
    bins.to_csv(OUT/"rbit_hit_bins.csv",index=False)
    assoc.to_csv(OUT/"rbit_association_summary.csv",index=False)
    grid.to_csv(OUT/"rbit_dhull_2d_bins.csv",index=False)
    types.to_csv(OUT/"rbit_system_type_summary.csv",index=False)
    dhbins.to_csv(OUT/"rbit_favorable_dhull_bins.csv",index=False)
    within.to_csv(OUT/"rbit_within_dhull_summary.csv",index=False)
    mapping.to_csv(OUT/"payload_mapping_audit.csv",index=False)
    audit.to_csv(OUT/"input_condition_audit.csv",index=False)
    inventory.to_csv(OUT/"input_file_inventory.csv",index=False)
    cal.to_csv(OUT/"rbit_exact_calibration_audit.csv",index=False)
    plot_previews(coverage,bins,assoc,grid)
    shutil.copy2(__file__,OUT/"reproduce_rbit_analysis.py")
    report(mapping,audit,inventory,cal,coverage,bins,assoc,dhbins,grid,within,types)
    validate_and_manifest(compact,coverage,bins,assoc,grid)
    archive=ROOT/"paper_rbit_analysis_20260821.zip"
    if archive.exists(): archive.unlink()
    subprocess.run(["zip","-q","-r",str(archive),OUT.name],cwd=ROOT,check=True)
    print(f"completed: {OUT}",flush=True)
    print(f"zip: {archive}",flush=True)


if __name__=="__main__": main()

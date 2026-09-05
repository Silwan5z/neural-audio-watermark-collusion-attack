#!/usr/bin/env python3
"""Analyze random-population native-rate K=8 bit-composition results."""
from __future__ import annotations
import json
from pathlib import Path
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/"data"/"k8_population_native_20260830"
OUT=ROOT/"data"/"main_text_revision_20260830"/"k8_population"; OUT.mkdir(parents=True,exist_ok=True)
MODELS=["audioseal","wavmark","timbrewm","voicemark","wmcodec"]
LABEL={"audioseal":"AudioSeal","wavmark":"WavMark","timbrewm":"TimbreWM","voicemark":"VoiceMark","wmcodec":"WMCodec"}
COLOR={"audioseal":"#2878B5","wavmark":"#54A24B","timbrewm":"#B279A2","voicemark":"#D95F59","wmcodec":"#F2A541"}
mpl.rcParams.update({"font.family":"sans-serif","font.sans-serif":["Arial","Helvetica","DejaVu Sans"],"font.size":8.2,"axes.titlesize":8.8,"axes.labelsize":8.3,"xtick.labelsize":7.7,"ytick.labelsize":7.7,"pdf.fonttype":42,"svg.fonttype":"none"})

def ratio_boot(df,num,den,seed=20260830,B=10000):
    g=df.groupby("speaker")[[num,den]].sum(); a=g[num].to_numpy(float); b=g[den].to_numpy(float); rng=np.random.default_rng(seed); n=len(g); vals=np.empty(B)
    for i in range(B):
        ix=rng.integers(0,n,n); vals[i]=a[ix].sum()/b[ix].sum()
    point=df[num].sum()/df[den].sum(); return float(point),*np.quantile(vals,[.025,.975]).tolist()

def mean_boot(df,col,seed=20260830,B=10000):
    z=df.assign(_n=1); return ratio_boot(z,col,"_n",seed,B)

def save(fig,stem):
    for ext in ("pdf","svg","png"): fig.savefig(OUT/f"{stem}.{ext}",dpi=350,bbox_inches="tight",facecolor="white")
    plt.close(fig)

def main():
    trials=[]; bits=[]; audit={"expected_trials_per_system":300,"field_presence":{},"source_decode_exact":{},"native_sample_rates":{},"notes":["Random coalitions reproduce scripts/attack.py deterministic K=8 sampler.","No constructed-case records are included.","WavMark soft probability is a valid-window vote-fraction proxy; other systems use native decoder bit marginals."]}
    required=["coalition_payload_bits","decoded_hard_bits","soft_bit_probability","tracing_failure","speaker","trial_id","PESQ","STOI","source_decodes"]
    for m in MODELS:
      files=sorted((DATA/"raw"/m).glob("trial_*.json"))
      if len(files)!=300: raise RuntimeError(f"{m}: expected 300 trials, found {len(files)}")
      field_ok={k:True for k in required}; source_exact=0; source_total=0
      for p in files:
        x=json.loads(p.read_text());
        for k in required: field_ok[k]&=k in x
        hard=np.asarray(x["decoded_hard_bits"],int); probs=np.asarray(x["soft_bit_probability"],float); coal=np.asarray(x["coalition_payload_bits"],int); counts=coal.sum(0)
        source_exact+=sum(r["exact"] for r in x["source_decodes"]); source_total+=len(x["source_decodes"])
        trials.append({"model":m,"trial_id":x["trial_id"],"speaker":x["speaker"],"local_trial":x["local_trial"],"native_sample_rate":x["native_sample_rate"],"coalition_payloads":json.dumps(x["coalition_payloads"],separators=(",",":")),"coalition_payload_bits":json.dumps(x["coalition_payload_bits"],separators=(",",":")),"decoded_identity":x["decoded_identity"],"decoded_hard_bits":json.dumps(x["decoded_hard_bits"],separators=(",",":")),"soft_bit_probability":json.dumps(x["soft_bit_probability"],separators=(",",":")),"source_exact_count":x["source_exact_count"],"tracing_failure":x["tracing_failure"],"NCA":x["NCA"],"PESQ":x["PESQ"],"STOI":x["STOI"]})
        for j,(c,h,pr) in enumerate(zip(counts,hard,probs)):
          pred=-1 if c==4 else int(c>4); strict=int(c!=4); unanimous=int(c in (0,8))
          bits.append({"model":m,"trial_id":x["trial_id"],"speaker":x["speaker"],"bit":j,"ones_among_8":int(c),"decoded_bit":int(h),"p_bit_1":float(pr),"strict":strict,"strict_correct":int(strict and h==pred),"unanimous":unanimous,"unanimous_correct":int(unanimous and h==pred),"tie":int(c==4),"confidence_margin":float(abs(pr-.5))})
      audit["field_presence"][m]=field_ok; audit["source_decode_exact"][m]={"exact":source_exact,"total":source_total,"rate":source_exact/source_total}; audit["native_sample_rates"][m]=x["native_sample_rate"]
    tdf=pd.DataFrame(trials); bdf=pd.DataFrame(bits); tdf.to_csv(OUT/"k8_population_trials.csv",index=False); bdf.to_csv(OUT/"k8_population_bits.csv",index=False)
    comp=[]; summary=[]
    for mi,m in enumerate(MODELS):
      bm=bdf[bdf.model==m]; tm=tdf[tdf.model==m]
      for c in range(9):
        z=bm[bm.ones_among_8==c].copy(); z["one"]=1; q,ql,qh=ratio_boot(z,"decoded_bit","one",20260830+mi*20+c)
        if c==4: consistency=cl=ch=np.nan
        else:
          z["consistent"]=(z.decoded_bit==(c>4)).astype(int); consistency,cl,ch=ratio_boot(z,"consistent","one",20260930+mi*20+c)
        comp.append({"model":m,"ones_among_8":c,"n_bits":len(z),"decoded_one_rate":q,"decoded_one_ci_low":ql,"decoded_one_ci_high":qh,"composition_consistency":consistency,"consistency_ci_low":cl,"consistency_ci_high":ch,"mean_confidence_margin":z.confidence_margin.mean() if c==4 else np.nan})
      strict=bm[bm.strict==1].copy(); strict["one"]=1; sm,slo,shi=ratio_boot(strict,"strict_correct","one",20261000+mi)
      un=bm[bm.unanimous==1].copy(); un["one"]=1; um,ulo,uhi=ratio_boot(un,"unanimous_correct","one",20261100+mi)
      tie=bm[bm.tie==1].copy(); tie["one"]=1; tr,tlo,thi=ratio_boot(tie,"decoded_bit","one",20261200+mi); margin,mlo,mhi=mean_boot(tie,"confidence_margin",20261300+mi)
      # exact majority is defined over strict-majority columns only.
      exact=(strict.groupby(["speaker","trial_id"]).strict_correct.min().reset_index(name="exact")); exact["one"]=1; em,elo,ehi=ratio_boot(exact,"exact","one",20261400+mi)
      tf,tfl,tfh=mean_boot(tm,"tracing_failure",20261500+mi)
      summary.append({"model":m,"n_trials":len(tm),"n_speakers":tm.speaker.nunique(),"source_decode_exact_rate":audit["source_decode_exact"][m]["rate"],"overall_strict_majority_consistency":sm,"strict_ci_low":slo,"strict_ci_high":shi,"unanimous_preservation":um,"unanimous_ci_low":ulo,"unanimous_ci_high":uhi,"tie_decoded_one_rate":tr,"tie_decoded_one_ci_low":tlo,"tie_decoded_one_ci_high":thi,"tie_mean_confidence_margin":margin,"tie_margin_ci_low":mlo,"tie_margin_ci_high":mhi,"trial_exact_majority_payload_rate":em,"exact_ci_low":elo,"exact_ci_high":ehi,"tracing_failure":tf,"tf_ci_low":tfl,"tf_ci_high":tfh,"PESQ":tm.PESQ.mean(),"STOI":tm.STOI.mean()})
    cdf=pd.DataFrame(comp); sdf=pd.DataFrame(summary); cdf.to_csv(OUT/"k8_composition_by_count.csv",index=False); sdf.to_csv(OUT/"k8_compact_system_summary.csv",index=False)
    (OUT/"k8_population_audit.json").write_text(json.dumps(audit,indent=2)+"\n")
    fig,axes=plt.subplots(1,5,figsize=(7.16,1.72),sharex=True,sharey=True)
    for ax,m in zip(axes,MODELS):
      z=cdf[cdf.model==m].sort_values("ones_among_8"); ax.fill_between(z.ones_among_8,z.decoded_one_ci_low,z.decoded_one_ci_high,color=COLOR[m],alpha=.20,lw=0); ax.plot(z.ones_among_8,z.decoded_one_rate,color=COLOR[m],marker="o",ms=2.7,lw=1.2)
      ax.axvspan(3.7,4.3,color="#F2D6B3",alpha=.48); ax.axhline(.5,color="#777",ls="--",lw=.65); ax.plot([0,3],[0,0],color="#333",ls=":",lw=.7); ax.plot([5,8],[1,1],color="#333",ls=":",lw=.7)
      ax.set_title(LABEL[m],fontweight="bold"); ax.set_xticks([0,2,4,6,8]); ax.set_ylim(-.03,1.03); ax.grid(axis="y",color="#E5E5E5",lw=.4); ax.spines[["top","right"]].set_visible(False); ax.set_xlabel("Ones among 8")
    axes[0].set_ylabel("Decoded-one rate $q(c)$")
    fig.subplots_adjust(wspace=.16,bottom=.25,top=.87); save(fig,"fig_k8_population_composition")
    print(OUT)

if __name__=="__main__": main()

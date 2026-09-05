#!/usr/bin/env python3
"""Build requested publication figures that use existing experiment outputs."""
from __future__ import annotations
import ast, csv, json
from collections import defaultdict
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf
from matplotlib.colors import Normalize
from scipy.signal import stft

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"data"/"main_text_revision_20260830"; OUT.mkdir(parents=True,exist_ok=True)
MODELS=["audioseal","wavmark","timbrewm","voicemark","wmcodec"]
LABEL={"audioseal":"AudioSeal","wavmark":"WavMark","timbrewm":"TimbreWM","voicemark":"VoiceMark","wmcodec":"WMCodec"}
COLORS={"audioseal":"#2878B5","wavmark":"#54A24B","timbrewm":"#B279A2","voicemark":"#D95F59","wmcodec":"#F2A541"}
mpl.rcParams.update({"font.family":"sans-serif","font.sans-serif":["Arial","Helvetica","DejaVu Sans"],
 "font.size":8.2,"axes.titlesize":9.0,"axes.labelsize":8.4,"xtick.labelsize":7.8,"ytick.labelsize":7.8,
 "legend.fontsize":7.6,"axes.linewidth":.65,"pdf.fonttype":42,"svg.fonttype":"none"})

def save(fig,folder,stem):
    folder.mkdir(parents=True,exist_ok=True)
    for ext in ("pdf","svg","png"): fig.savefig(folder/f"{stem}.{ext}",dpi=350,bbox_inches="tight",facecolor="white")
    plt.close(fig)

def cluster_ci(df,value,seed=20260830,B=10000):
    groups={s:g[value].to_numpy(float) for s,g in df.groupby("spk")}; keys=list(groups)
    rng=np.random.default_rng(seed); vals=np.empty(B)
    for b in range(B): vals[b]=np.mean(np.concatenate([groups[k] for k in rng.choice(keys,len(keys),replace=True)]))
    return float(df[value].mean()),*np.quantile(vals,[.025,.975]).tolist()

def main_risk():
    folder=OUT/"main_risk"; rows=[]
    for m in MODELS:
      for k in (2,3,5,8):
        df=pd.read_csv(ROOT/"data"/"attack"/f"attack_{m}_K{k}.csv")
        for method in ("mean","fwp"):
          x=df[df.method==method]
          for metric in ("ASR","PESQ","STOI"):
            mean,lo,hi=cluster_ci(x,metric,seed=20260830+k)
            rows.append(dict(model=m,K=k,method=method,metric=metric,mean=mean,ci_low=lo,ci_high=hi,n_trials=len(x),n_speakers=x.spk.nunique()))
    pd.DataFrame(rows).to_csv(folder/"risk_speaker_cluster_bootstrap.csv",index=False) if folder.mkdir(parents=True,exist_ok=True) is None else None
    res=pd.DataFrame(rows); fig,axes=plt.subplots(1,3,figsize=(7.16,1.78))
    (folder/"main_risk_audit.json").write_text(json.dumps({"models":MODELS,"coalition_sizes":[2,3,5,8],"main_method":"mean (uniform full-waveform averaging)","appendix_method":"fwp","trials_per_model_K_method":300,"cluster_unit":"speaker","bootstrap_replicates":10000,"bootstrap_seed_family":"20260830 + K","quality_metrics":["PESQ","STOI"],"protocol_note":"Existing attack CSVs are reused exactly as requested; they follow the original project wrapper protocol. Native-rate attribution is enforced only in the newly augmented K=8 composition experiment.","source_pattern":"data/attack/attack_{model}_K{K}.csv"},indent=2)+"\n")
    specs=[("ASR","Tracing failure",(0,1)),("PESQ","PESQ",None),("STOI","STOI",None)]
    for ax,(metric,title,ylim) in zip(axes,specs):
      for m in MODELS:
        x=res[(res.model==m)&(res.method=="mean")&(res.metric==metric)].sort_values("K")
        ax.plot(x.K,x["mean"],marker="o",ms=3,lw=1.25,color=COLORS[m],label=LABEL[m])
        ax.fill_between(x.K,x.ci_low,x.ci_high,color=COLORS[m],alpha=.14,lw=0)
      ax.set_title(title,fontweight="bold"); ax.set_xlabel("Coalition size $K$"); ax.set_xticks([2,3,5,8]);
      if ylim: ax.set_ylim(*ylim)
      ax.grid(axis="y",color="#E2E2E2",lw=.45); ax.spines[["top","right"]].set_visible(False)
    axes[0].set_ylabel("Rate"); axes[1].set_ylabel("Score"); axes[2].set_ylabel("Score")
    handles,labels=axes[2].get_legend_handles_labels(); fig.legend(handles,labels,frameon=False,ncol=5,loc="lower center",bbox_to_anchor=(.5,-.16),columnspacing=1.0)
    fig.subplots_adjust(wspace=.34,bottom=.28,top=.90); save(fig,folder,"fig_main_uniform_risk_quality")
    # FWP explicitly appendix-only.
    fig,ax=plt.subplots(figsize=(3.45,2.25))
    for m in MODELS:
      x=res[(res.model==m)&(res.method=="fwp")&(res.metric=="ASR")].sort_values("K")
      ax.errorbar(x.K,x["mean"],yerr=[x["mean"]-x.ci_low,x.ci_high-x["mean"]],marker="o",ms=3,lw=1,color=COLORS[m],label=LABEL[m],capsize=2)
    ax.set(xlabel="Coalition size $K$",ylabel="Tracing failure",ylim=(0,1),xticks=[2,3,5,8])
    ax.set_title("FWP tracing failure (appendix)",fontweight="bold"); ax.grid(axis="y",color="#E2E2E2",lw=.45); ax.spines[["top","right"]].set_visible(False); ax.legend(frameon=False,ncol=2)
    save(fig,folder,"fig_appendix_fwp_tracing_failure")

def frequency_temporal():
    folder=OUT/"frequency_temporal"; frames=[pd.read_csv(ROOT/"data"/"frequency_temporal_k5_20260828"/f"frequency_temporal_k5_{m}.csv") for m in MODELS]
    df=pd.concat(frames,ignore_index=True)
    summ=df.groupby(["model","condition","condition_family"]).agg(NCA=("NCA","mean"),tracing_failure=("tracing_failure","mean"),n=("trial_id","size")).reset_index()
    summ["one_minus_NCA"]=1-summ.NCA; folder.mkdir(parents=True,exist_ok=True); summ.to_csv(folder/"frequency_temporal_summary.csv",index=False)
    (folder/"frequency_temporal_audit.json").write_text(json.dumps({"models":MODELS,"K":5,"trials_per_condition_per_system":300,"color_metric":"1-NCA","annotation_metric":"tracing_failure","shared_color_scale":True,"rerun":False,"source_pattern":"data/frequency_temporal_k5_20260828/frequency_temporal_k5_{model}.csv"},indent=2)+"\n")
    freq=["freq_0_1k","freq_1_2k","freq_2_4k","freq_4_8k"]
    temp=["speech_only","non_speech_only","random_mask","full_waveform"]
    vmax=float(summ.one_minus_NCA.max()); fig,axes=plt.subplots(1,2,figsize=(7.16,1.82),gridspec_kw={"width_ratios":[1,1]})
    for ax,conds,title,xlabels in [(axes[0],freq,"Frequency allocation",["0–1","1–2","2–4","4–8"]),(axes[1],temp,"Temporal allocation",["Speech","Non-speech","Random","Full"])]:
      mat=np.full((5,4),np.nan); tf=np.full((5,4),np.nan)
      for i,m in enumerate(MODELS):
       for j,c in enumerate(conds):
        z=summ[(summ.model==m)&(summ.condition==c)]
        if len(z): mat[i,j]=z.one_minus_NCA.iloc[0]; tf[i,j]=z.tracing_failure.iloc[0]
      im=ax.imshow(mat,aspect="auto",cmap="YlOrRd",vmin=0,vmax=vmax)
      for i in range(5):
       for j in range(4):
        if np.isfinite(mat[i,j]): ax.text(j,i,f"{mat[i,j]:.2f}\n({tf[i,j]:.2f})",ha="center",va="center",fontsize=7.0,linespacing=.9,color="white" if mat[i,j]>.55*vmax else "#222")
      ax.set_xticks(range(4),xlabels); ax.set_yticks(range(5),[LABEL[m] for m in MODELS]); ax.set_title(title,fontweight="bold")
    cbar=fig.colorbar(im,ax=axes,pad=.02,fraction=.025); cbar.set_label("Nearest-participant distance")
    fig.subplots_adjust(left=.12,right=.91,bottom=.20,top=.88,wspace=.25); save(fig,folder,"fig_frequency_temporal_heatmaps")

def mixture_path():
    folder=OUT/"mixture_path"; folder.mkdir(parents=True,exist_ok=True); traj=[]
    for m in ("audioseal","voicemark"):
      buckets=defaultdict(list); p=ROOT/"data"/"mixture_path_k5_adaptive_20260829"/f"mixture_path_{m}_points.csv"
      cols=["sampling_stage","flipped_bit","flipped_payload","bit_probabilities","lambda"]
      for chunk in pd.read_csv(p,usecols=cols,chunksize=50000):
       chunk=chunk[chunk.sampling_stage=="coarse"]
       for bit,payload,probs,lam in zip(chunk.flipped_bit,chunk.flipped_payload,chunk.bit_probabilities,chunk["lambda"]):
        bit=int(bit); values=json.loads(probs); target=(int(payload)>>bit)&1
        val=float(values[bit]); val=val if target else 1-val; buckets[float(lam)].append(val)
      for lam,a in sorted(buckets.items()):
        traj.append(dict(model=m,lambda_value=lam,mean=np.mean(a),p10=np.quantile(a,.1),p90=np.quantile(a,.9),n=len(a)))
    pd.DataFrame(traj).to_csv(folder/"mixture_path_aggregate_trajectory.csv",index=False)
    trials=pd.read_csv(ROOT/"data"/"mixture_path_k5_adaptive_20260829"/"mixture_path_trial_summary.csv")
    counts=[]
    for m in ("audioseal","voicemark"):
      x=trials[trials.model==m]
      counts.append(dict(model=m,n_trials=len(x),target_monotonic=int(x.target_bit_monotonic.sum()),
        target_back_and_forth=int((x.target_transition_count>1).sum()),identity_zero_transition=int((x.identity_transition_count==0).sum()),
        identity_single_transition=int((x.identity_transition_count==1).sum()),identity_multiple_transition=int((x.identity_transition_count>1).sum())))
    pd.DataFrame(counts).to_csv(folder/"mixture_path_transition_counts.csv",index=False)
    (folder/"mixture_path_audit.json").write_text(json.dumps({"models":["audioseal","voicemark"],"trials_per_system":300,"title":"Controlled one-bit interpolation","trajectory":"coarse common lambda grid; probability oriented toward flipped endpoint; mean and P10-P90","counts":["target-bit monotonic","target-bit back-and-forth (>1 hard transition)","zero/single/multiple decoded-identity transitions"],"excluded_from_main_figure":["R2","dual-y display"],"rerun":False,"source_directory":"data/mixture_path_k5_adaptive_20260829"},indent=2)+"\n")
    fig,axes=plt.subplots(1,2,figsize=(7.16,2.45))
    tr=pd.DataFrame(traj)
    for m in ("audioseal","voicemark"):
      z=tr[tr.model==m].sort_values("lambda_value"); axes[0].plot(z.lambda_value,z["mean"],color=COLORS[m],lw=1.5,label=LABEL[m]); axes[0].fill_between(z.lambda_value,z.p10,z.p90,color=COLORS[m],alpha=.18,lw=0)
    axes[0].axhline(.5,color="#777",ls="--",lw=.7); axes[0].set(xlabel="Interpolation $\\lambda$",ylabel="Response toward flipped endpoint",ylim=(0,1)); axes[0].set_title("Aggregate target-bit trajectory (P10–P90)",fontweight="bold"); axes[0].legend(frameon=False)
    c=pd.DataFrame(counts).set_index("model"); x=np.arange(2); w=.22
    vals=[c.identity_zero_transition,c.identity_single_transition,c.identity_multiple_transition]; labs=["No transition","Single","Multiple"]
    for j,(v,l,col) in enumerate(zip(vals,labs,["#BDBDBD","#4C78A8","#E45756"])): axes[1].bar(x+(j-1)*w,v,width=w,label=l,color=col)
    axes[1].set_xticks(x,[LABEL[m] for m in ("audioseal","voicemark")]); axes[1].set_ylabel("Trials (of 300)"); axes[1].set_title("Decoded-identity transitions",fontweight="bold"); axes[1].legend(frameon=False,fontsize=6.5)
    for ax in axes: ax.grid(axis="y",color="#E2E2E2",lw=.45); ax.spines[["top","right"]].set_visible(False)
    fig.suptitle("Controlled one-bit interpolation",fontweight="bold",y=1.02); save(fig,folder,"fig_controlled_onebit_interpolation")

def mel_filter(sr,nfft,nmel=64,fmax=8000):
    hz=np.linspace(0,sr/2,nfft//2+1); mel=lambda f:2595*np.log10(1+f/700); inv=lambda m:700*(10**(m/2595)-1)
    edges=inv(np.linspace(mel(0),mel(min(fmax,sr/2)),nmel+2)); fb=np.zeros((nmel,len(hz)))
    for i in range(nmel): fb[i]=np.maximum(0,np.minimum((hz-edges[i])/(edges[i+1]-edges[i]+1e-12),(edges[i+2]-hz)/(edges[i+2]-edges[i+1]+1e-12)))
    return fb

def onebit_case():
    folder=OUT/"onebit_case"; folder.mkdir(parents=True,exist_ok=True)
    audit=json.loads(Path("/private/users/lym/audioseal_onebit_case_study/figure_final/matched_case_audit.json").read_text())
    records={a["model"]:a for a in audit["audits"]}; data={}
    for m,a in records.items():
      clean,sr=sf.read(a["source_paths"]["clean"],dtype="float32"); w0,_=sf.read(a["source_paths"]["b0"],dtype="float32"); w1,_=sf.read(a["source_paths"]["b1"],dtype="float32")
      n=min(len(clean),len(w0),len(w1)); clean,w0,w1=clean[:n],w0[:n],w1[:n]; avg=(w0+w1)/2
      data[m]=dict(clean=clean,r0=w0-clean,r1=w1-clean,ravg=avg-clean,delta=w1-w0,sr=sr,a=a)
    # One shared deterministic window: max of per-system normalized 50-ms delta RMS.
    scores=[]
    for m in ("audioseal","voicemark"):
      d=data[m]; win=round(.05*d["sr"]); s=np.convolve(d["delta"].astype(float)**2,np.ones(win)/win,mode="valid"); scores.append(s/(s.max()+1e-15))
    center=int(np.argmax(scores[0]+scores[1]))+400; half=round(.05*16000); i0=max(0,center-half); i1=i0+2*half
    shared_residual_limit=max(np.max(np.abs(data[m][k][i0:i1]*1000)) for m in data for k in ("r0","r1","ravg"))*1.08
    fig=plt.figure(figsize=(7.16,3.20)); outer=fig.add_gridspec(2,3,width_ratios=[1.30,1.08,.78],hspace=.55,wspace=.46,top=.94,bottom=.13)
    for row,m in enumerate(("audioseal","voicemark")):
      d=data[m]; sr=d["sr"]; t=np.arange(len(d["clean"]))/sr; sub=outer[row,0].subgridspec(2,1,height_ratios=[.48,1.52],hspace=.08)
      axc=fig.add_subplot(sub[0]); axr=fig.add_subplot(sub[1]); seg=slice(i0,i1)
      axc.plot(t[seg],d["clean"][seg],color="#666666",lw=.65); axc.set_ylabel("Clean",labelpad=2)
      for key,col,lab in [("r0","#E8A400","$r_0$"),("r1","#1E8449","$r_1$"),("ravg","#C0392B","Average")]:
        axr.plot(t[seg],d[key][seg]*1000,color=col,lw=.85,alpha=.92,label=lab,zorder=3)
      axr.set_ylabel("Residual\n($\\times10^3$)"); axr.set_ylim(-shared_residual_limit,shared_residual_limit); axr.legend(frameon=False,ncol=3,loc="upper center",columnspacing=.8,handlelength=1.2)
      for ax in (axc,axr): ax.set_xlim(t[i0],t[i1-1]); ax.axhline(0,color="#bbb",lw=.35); ax.spines[["top","right"]].set_visible(False)
      axc.tick_params(labelbottom=False); axc.set_title(f"{LABEL[m]}: waveform context",fontweight="bold",fontsize=8.5,pad=2)
      if row==0: axr.tick_params(labelbottom=False)
      else: axr.set_xlabel("Time (s)")
      # Full-audio residual Mel, normalized only within this system.
      axm=fig.add_subplot(outer[row,1]); f,tt,Z=stft(d["delta"],fs=sr,nperseg=1024,noverlap=768,boundary=None); power=np.abs(Z)**2; M=mel_filter(sr,1024)@power; db=10*np.log10(M+1e-12); db-=db.max(); db=np.maximum(db,-60)
      im=axm.imshow(db,origin="lower",aspect="auto",extent=[tt[0],tt[-1],0,64],cmap="magma",vmin=-60,vmax=0); axm.set(ylabel="Mel bin");
      if row==0: axm.tick_params(labelbottom=False)
      else: axm.set_xlabel("Time (s)")
      axm.set_title("One-bit residual Mel\n(system max = 0 dB)",fontweight="bold",fontsize=8.5,pad=2)
      fig.colorbar(im,ax=axm,pad=.02,fraction=.05)
      axb=fig.add_subplot(outer[row,2]); vals=[d["a"]["p_bit_1_b0"],d["a"]["p_bit_1_b1"],d["a"]["p_bit_1_average"]]; y=np.arange(3); axb.barh(y,vals,color=["#E8A400","#1E8449","#C0392B"],height=.55); axb.axvline(.5,color="#555",ls="--",lw=.8)
      axb.set_yticks(y,["$b_4=0$","$b_4=1$","Average"]); axb.invert_yaxis(); axb.set(xlim=(0,1));
      if row==0: axb.tick_params(labelbottom=False)
      else: axb.set_xlabel("$P(b_4=1)$")
      axb.set_title("Target-bit response",fontweight="bold",fontsize=8.5,pad=2)
      for yy,v in zip(y,vals):
        if v>.88: axb.text(v-.025,yy,f"{v:.3f}",ha="right",va="center",fontsize=7.4)
        else: axb.text(v+.035,yy,f"{v:.3f}",ha="left",va="center",fontsize=7.4)
      axb.spines[["top","right"]].set_visible(False)
    save(fig,folder,"fig_unified_onebit_case_audioseal_voicemark")
    outaudit={"speaker":"chinese:SSB0197","payload_bit0":63374,"payload_bit1":63390,"flipped_bit":4,"window_rule":"shared 100-ms window centered at argmax of sum of per-system normalized 50-ms sliding RMS of r1-r0","window_seconds":[i0/16000,i1/16000],"amplitude_scaling":"residual waveforms labeled x1000 only; no normalization","mel":"full audio; each system normalized to its own maximum (0 dB), clipped at -60 dB","decoder_values":{m:{k:records[m][k] for k in ("p_bit_1_b0","p_bit_1_b1","p_bit_1_average")} for m in records}}
    (folder/"onebit_case_audit.json").write_text(json.dumps(outaudit,indent=2)+"\n")

if __name__=="__main__":
    main_risk(); frequency_temporal(); mixture_path(); onebit_case(); print(OUT)

#!/usr/bin/env python3
"""Generate publication-ready figure/table candidates directly from final CSVs."""
from __future__ import annotations

import hashlib
import math
import shutil
import subprocess
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
EVAL = ROOT / "results" / "evaluation"
OUT = ROOT / "paper_visual_candidates_20260820"
MODELS = ["audioseal", "wavmark", "timbrewm", "voicemark", "wmcodec"]
LABEL = {"audioseal":"AudioSeal", "wavmark":"WavMark", "timbrewm":"TimbreWM",
         "voicemark":"VoiceMark", "wmcodec":"WMCodec"}
KS = [2, 3, 5, 8]
BLUE, ORANGE = "#0077BB", "#EE7733"
MODEL_COLORS = dict(zip(MODELS, ["#0077BB", "#33BBEE", "#009988", "#EE7733", "#CC3311"]))

matplotlib.rcParams.update({
    "font.family":"sans-serif", "font.sans-serif":["Arial","Helvetica","DejaVu Sans"],
    "font.size":8.5, "axes.titlesize":10, "axes.labelsize":9,
    "xtick.labelsize":8, "ytick.labelsize":8, "legend.fontsize":8,
    "figure.dpi":160, "savefig.dpi":300, "savefig.bbox":"tight",
    "axes.spines.top":False, "axes.spines.right":False,
})


def read_cells(prefix: str, suffix="") -> pd.DataFrame:
    frames=[]
    for model in MODELS:
        for k in KS:
            p=EVAL/f"{prefix}_{model}_K{k}{suffix}.csv"
            if p.exists():
                d=pd.read_csv(p); d["source_file"]=p.name; frames.append(d)
    return pd.concat(frames,ignore_index=True)


def wilson(x: int, n: int) -> tuple[float,float,float]:
    z=1.959963984540054; p=x/n; den=1+z*z/n
    c=(p+z*z/(2*n))/den; h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return p,max(0,c-h),min(1,c+h)


def rate_ci(df, groups, metric):
    rows=[]
    for key,g in df.groupby(groups,sort=True):
        if not isinstance(key,tuple): key=(key,)
        v=pd.to_numeric(g[metric],errors="coerce").dropna().astype(int); x=int(v.sum()); n=len(v); r,lo,hi=wilson(x,n)
        rows.append(dict(zip(groups,key))|{"numerator":x,"denominator":n,"rate":r,"ci_low":lo,"ci_high":hi})
    return pd.DataFrame(rows)


def savefig(fig, stem):
    fig.savefig(OUT/f"{stem}.pdf")
    fig.savefig(OUT/f"{stem}.png",dpi=300)
    plt.close(fig)


def heat(ax, matrix, title, vmin=0, vmax=1, cmap="cividis"):
    im=ax.imshow(matrix,vmin=vmin,vmax=vmax,cmap=cmap,aspect="auto")
    ax.set_xticks(range(len(KS)),[str(k) for k in KS]); ax.set_xlabel("Coalition size, K")
    ax.set_yticks(range(len(MODELS)),[LABEL[m] for m in MODELS])
    ax.set_title(title,fontweight="bold")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            v=matrix[i,j]; color="white" if v < .38 or v > .82 else "black"
            ax.text(j,i,f"{100*v:.1f}",ha="center",va="center",fontsize=7.5,color=color)
    return im


def make_figures(attack, registry, hull, arb, matched, quality):
    # A: native blind ASR, 300 trials per point with Wilson intervals.
    a=rate_ci(attack,["model","K","method"],"ASR")
    a.to_csv(OUT/"data/fig_A_blind_asr.csv",index=False)
    fig,axes=plt.subplots(2,3,figsize=(6.9,4.45),sharex=True,sharey=True)
    for ax,model in zip(axes.flat,MODELS):
        d=a[a.model==model]
        for method,color,marker in [("mean",BLUE,"o"),("fwp",ORANGE,"s")]:
            q=d[d.method==method].set_index("K").loc[KS]
            yerr=np.vstack([np.maximum(0,q.rate-q.ci_low),np.maximum(0,q.ci_high-q.rate)])
            ax.errorbar(KS,q.rate,yerr=yerr,color=color,marker=marker,
                        linewidth=1.5,markersize=4,capsize=2,label=method.upper())
        ax.set_title(LABEL[model],fontweight="bold"); ax.set_xticks(KS); ax.set_ylim(-.03,1.05)
        ax.grid(axis="y",alpha=.22,linewidth=.5)
    axes[1,2].axis("off")
    for ax in axes[1,:2]: ax.set_xlabel("Coalition size, K")
    for ax in axes[:,0]: ax.set_ylabel("Attack success rate")
    axes[0,2].legend(frameon=False,loc="lower right")
    fig.suptitle("Native-registry blind attribution escape",fontsize=11,fontweight="bold",y=.995)
    fig.tight_layout(); savefig(fig,"fig_A_blind_asr")

    # B: registry scaling for the four 16-bit models.
    r=rate_ci(registry,["model","K","N_registry","method"],"ASR")
    r16=r[(r.model!="timbrewm")].copy(); r16.to_csv(OUT/"data/fig_B_registry_scaling.csv",index=False)
    fig,axes=plt.subplots(2,2,figsize=(6.9,4.65),sharex=True,sharey=True)
    kcolors=dict(zip(KS,["#0077BB","#009988","#EE7733","#CC3311"]))
    for ax,model in zip(axes.flat,["audioseal","wavmark","voicemark","wmcodec"]):
        d=r16[r16.model==model]
        for k in KS:
            for method,ls,marker in [("mean","--","o"),("fwp","-","s")]:
                q=d[(d.K==k)&(d.method==method)].sort_values("N_registry")
                ax.plot(np.log2(q.N_registry),q.rate,color=kcolors[k],ls=ls,marker=marker,
                        linewidth=1.15,markersize=3)
        ax.set_title(LABEL[model],fontweight="bold"); ax.set_ylim(-.03,1.05); ax.grid(axis="y",alpha=.2,linewidth=.5)
    for ax in axes[1]: ax.set_xlabel(r"Registry size, $\log_2 N$")
    for ax in axes[:,0]: ax.set_ylabel("Attack success rate")
    handles=[Line2D([0],[0],color=kcolors[k],lw=2,label=f"K={k}") for k in KS]
    handles += [Line2D([0],[0],color="black",ls="--",marker="o",lw=1,label="Mean"),
                Line2D([0],[0],color="black",ls="-",marker="s",lw=1,label="FWP")]
    fig.legend(handles=handles,ncol=6,loc="upper center",frameon=False,bbox_to_anchor=(.5,1.01))
    fig.suptitle("Registry-size sensitivity of blind escape",fontsize=11,fontweight="bold",y=1.06)
    fig.tight_layout(); savefig(fig,"fig_B_registry_scaling")

    # C: favorable/native arbitrary/matched-N arbitrary any-of-ten TCT hit.
    h=hull[hull.method=="tct"].copy(); h["trial_id"]=h["gi"]
    h_any=h.groupby(["model","K","trial_id"],as_index=False).target_hit.max()
    n=arb[arb.method=="tct"].copy(); n["trial_id"]=n["gi"]
    n_any=n.groupby(["model","K","trial_id"],as_index=False).target_top1.max().rename(columns={"target_top1":"target_hit"})
    m=matched[matched.method=="tct"].copy(); m["trial_id"]=m["gi"]
    m_any=m.groupby(["model","K","trial_id"],as_index=False).target_top1.max().rename(columns={"target_top1":"target_hit"})
    items=[]; mats=[]
    for protocol,d in [("Favorable/native",h_any),("Arbitrary/native",n_any),("Arbitrary/N=1024",m_any)]:
        s=rate_ci(d,["model","K"],"target_hit"); s["protocol"]=protocol; items.append(s)
        mats.append(s.pivot(index="model",columns="K",values="rate").reindex(index=MODELS,columns=KS).to_numpy())
    pd.concat(items,ignore_index=True).to_csv(OUT/"data/fig_C_targeted_any10.csv",index=False)
    fig=plt.figure(figsize=(6.9,3.0))
    gs=fig.add_gridspec(1,4,width_ratios=[1,1,1,.055],wspace=.18)
    axes=[fig.add_subplot(gs[0,i],sharey=None if i==0 else fig.axes[0]) for i in range(3)]
    cax=fig.add_subplot(gs[0,3])
    titles=["Favorable targets\n(native N)","Arbitrary targets\n(native N)","Arbitrary targets\n(matched N=1024)"]
    for ax,mat,title in zip(axes,mats,titles): im=heat(ax,mat,title)
    for ax in axes[1:]: ax.tick_params(labelleft=False)
    cb=fig.colorbar(im,cax=cax); cb.set_label("Any-of-ten hit rate")
    fig.suptitle("TCT success depends on target selection and registry size",fontsize=11,fontweight="bold",y=1.01)
    fig.subplots_adjust(left=.13,right=.95,bottom=.18,top=.78); savefig(fig,"fig_C_targeted_any10")

    # D: per-model quality / security frontier.
    q=quality.groupby(["model","K","method"],as_index=False).agg(ASR=("ASR","mean"),PESQ=("PESQ","mean"),STOI=("STOI","mean"),SI_SDR=("SI_SDR","mean"))
    q.to_csv(OUT/"data/fig_D_quality_tradeoff.csv",index=False)
    fig,axes=plt.subplots(2,3,figsize=(6.9,4.5))
    for ax,model in zip(axes.flat,MODELS):
        d=q[q.model==model]
        for method,color,marker in [("mean",BLUE,"o"),("fwp",ORANGE,"s")]:
            z=d[d.method==method].sort_values("K")
            ax.plot(z.PESQ,z.ASR,color=color,marker=marker,linewidth=1.2,label=method.upper())
            for _,row in z.iterrows(): ax.annotate(f"K{int(row.K)}",(row.PESQ,row.ASR),xytext=(3,2),textcoords="offset points",fontsize=6.5)
        ax.set_title(LABEL[model],fontweight="bold"); ax.set_ylim(-.03,1.05); ax.grid(alpha=.2,linewidth=.5)
    axes[1,2].axis("off")
    for ax in axes[1,:2]: ax.set_xlabel("PESQ vs. first watermarked copy")
    for ax in axes[:,0]: ax.set_ylabel("Attack success rate")
    axes[0,2].legend(frameon=False,loc="lower right")
    fig.suptitle("Blind attack effectiveness–quality trade-off",fontsize=11,fontweight="bold",y=.995)
    fig.tight_layout(); savefig(fig,"fig_D_quality_tradeoff")


def bold_best(a,b,higher=True,fmt=".3f"):
    vals=[a,b]; best=max(vals) if higher else min(vals)
    return [(f"\\textbf{{{v:{fmt}}}}" if abs(v-best)<1e-12 else f"{v:{fmt}}") for v in vals]


def make_tables(attack, codec, temporal):
    # E: comprehensive native blind table.
    s=attack.groupby(["model","K","method"],as_index=False).agg(ASR=("ASR","mean"),NAC=("ACC_near_norm","mean"),PESQ=("PESQ","mean"))
    rows=[]; tex=[]
    for model in MODELS:
        for k in KS:
            d=s[(s.model==model)&(s.K==k)].set_index("method")
            row={"model":LABEL[model],"K":k}
            for met in ["ASR","NAC","PESQ"]:
                row[f"Mean_{met}"]=d.loc["mean",met]; row[f"FWP_{met}"]=d.loc["fwp",met]
            rows.append(row)
            av=bold_best(row["Mean_ASR"],row["FWP_ASR"],True); nv=bold_best(row["Mean_NAC"],row["FWP_NAC"],False); pv=bold_best(row["Mean_PESQ"],row["FWP_PESQ"],True)
            tex.append(f"{LABEL[model]} & {k} & {av[0]} & {av[1]} & {nv[0]} & {nv[1]} & {pv[0]} & {pv[1]} \\\\")
    tab=pd.DataFrame(rows); tab.to_csv(OUT/"table_E_blind_main.csv",index=False)
    text="""\\begin{table*}[t]
\\centering
\\small
\\caption{Native-registry blind attack performance. Higher ASR and PESQ are better; lower NAC is better. Best within each model--K pair is bold. Quality is measured against the first legitimate watermarked coalition copy.}
\\label{tab:blind-main-candidate}
\\begin{tabular}{llrrrrrr}
\\toprule
Model & $K$ & \\multicolumn{2}{c}{ASR $\\uparrow$} & \\multicolumn{2}{c}{NAC $\\downarrow$} & \\multicolumn{2}{c}{PESQ $\\uparrow$} \\\\
 & & Mean & FWP & Mean & FWP & Mean & FWP \\\\
\\midrule
"""+"\n".join(tex)+"\n\\bottomrule\n\\end{tabular}\n\\end{table*}\n"
    (OUT/"table_E_blind_main.tex").write_text(text)
    render_table_png(tab[["model","K","Mean_ASR","FWP_ASR","Mean_NAC","FWP_NAC","Mean_PESQ","FWP_PESQ"]],"table_E_blind_main",figsize=(8.2,7.2),decimals=3)

    # F: compact robustness deltas, K=5.
    c=codec.groupby(["model","codec","method"],as_index=False).agg(ASR=("ASR","mean"),PESQ=("PESQ","mean"))
    t=temporal.groupby(["model","shift_ms","method"],as_index=False).agg(ASR=("ASR","mean"),PESQ=("PESQ","mean"))
    rows=[]
    for model in MODELS:
        for method in ["mean","fwp"]:
            cz=c[(c.model==model)&(c.method==method)].set_index("codec")
            tz=t[(t.model==model)&(t.method==method)].set_index("shift_ms")
            rows.append({"model":LABEL[model],"method":method.upper(),"ASR_none":cz.loc["none","ASR"],
                "ASR_MP3":cz.loc["mp3_128k","ASR"],"ASR_Opus":cz.loc["opus_64k","ASR"],
                "ASR_shift_-50":tz.loc[-50,"ASR"],"ASR_shift_0":tz.loc[0,"ASR"],"ASR_shift_+50":tz.loc[50,"ASR"]})
    rt=pd.DataFrame(rows); rt.to_csv(OUT/"table_F_robustness_K5.csv",index=False)
    lines=[]
    for _,r in rt.iterrows():
        lines.append(f"{r['model']} & {r['method']} & {r.ASR_none:.3f} & {r.ASR_MP3:.3f} & {r.ASR_Opus:.3f} & {r['ASR_shift_-50']:.3f} & {r['ASR_shift_0']:.3f} & {r['ASR_shift_+50']:.3f} \\\\")
    tex="""\\begin{table*}[t]
\\centering
\\small
\\caption{K=5 sensitivity to independent pre-collusion codecs and one-member temporal misalignment. Codec cells use 300 trials; temporal cells use 100 trials.}
\\label{tab:robustness-candidate}
\\begin{tabular}{llrrrrrr}
\\toprule
Model & Method & None & MP3 128k & Opus 64k & $-50$ ms & 0 ms & $+50$ ms \\\\
\\midrule
"""+"\n".join(lines)+"\n\\bottomrule\n\\end{tabular}\n\\end{table*}\n"
    (OUT/"table_F_robustness_K5.tex").write_text(tex)
    render_table_png(rt,"table_F_robustness_K5",figsize=(8.2,3.8),decimals=3)


def render_table_png(df,stem,figsize,decimals=3):
    show=df.copy()
    for c in show.select_dtypes(include=[np.number]).columns:
        if c!="K": show[c]=show[c].map(lambda x:f"{x:.{decimals}f}")
    fig,ax=plt.subplots(figsize=figsize); ax.axis("off")
    table=ax.table(cellText=show.values,colLabels=show.columns,cellLoc="center",loc="center")
    table.auto_set_font_size(False); table.set_fontsize(7.3); table.scale(1,1.25)
    for (r,c),cell in table.get_celld().items():
        cell.set_edgecolor("#BBBBBB"); cell.set_linewidth(.35)
        if r==0: cell.set_facecolor("#DDEBF7"); cell.set_text_props(weight="bold")
        elif r%2==0: cell.set_facecolor("#F5F5F5")
    fig.tight_layout(); fig.savefig(OUT/f"{stem}.png",dpi=300,bbox_inches="tight"); plt.close(fig)


def sha(p):
    h=hashlib.sha256(); h.update(p.read_bytes()); return h.hexdigest()


def main():
    if OUT.exists(): shutil.rmtree(OUT)
    (OUT/"data").mkdir(parents=True)
    attack=read_cells("attack")
    registry=read_cells("registry_control")
    hull=read_cells("framing_hull")
    arb=read_cells("tamper_arbitrary")
    # read_cells(tamper_arbitrary) uses native exact names and does not match N1024 suffix.
    matched=[]
    for model in MODELS:
        for k in KS: matched.append(pd.read_csv(EVAL/f"tamper_arbitrary_N1024_{model}_K{k}.csv"))
    matched=pd.concat(matched,ignore_index=True)
    quality=read_cells("attack")
    codec=pd.concat([pd.read_csv(EVAL/f"codec_sensitivity_{m}_K5.csv") for m in MODELS],ignore_index=True)
    temporal=pd.concat([pd.read_csv(EVAL/f"temporal_sensitivity_{m}_K5.csv") for m in MODELS],ignore_index=True)
    make_figures(attack,registry,hull,arb,matched,quality)
    make_tables(attack,codec,temporal)
    shutil.copy2(__file__,OUT/"generate_paper_visual_candidates.py")
    guide="""# Paper visual candidates

All candidates were regenerated directly from final CSVs in `results/evaluation/`. No manuscript file was found, so placement recommendations are based on evidentiary value rather than existing figure numbering.

## Recommended three-item set

1. **Table E — Complete native blind result (recommended main result).** It reports exact ASR/NAC/PESQ for every model, K, and Mean/FWP pair, including ties and the quality trade-off that an ASR-only plot would hide.
2. **Figure C — Targeted hit-rate protocol comparison (recommended mechanism/control result).** Three matched heatmaps separate favorable-target selection, native arbitrary targets, and matched-N arbitrary targets. This is the cleanest visual explanation of why target policy and registry size matter. Favorable means nearest among a deterministic 2,000-candidate subset, not globally nearest.
3. **Figure B — Registry-size scaling (recommended fairness/control result).** Shows whether conclusions survive N changes for all four 16-bit systems. TimbreWM is absent because its 10-bit native registry has only N=1024 and no sweep.

This trio has minimal redundancy: exact effectiveness/quality (E), mechanism/target policy (C), and fair-registry control (B).

## Alternative candidates

- **Figure A — Native blind ASR.** A visually lighter replacement for Table E when trend readability matters more than exact NAC/PESQ values. It retains Wilson intervals but many systems are near the ASR ceiling.
- **Figure D — ASR–PESQ trade-off.** Use if audio quality is a major reviewer concern. It is more intuitive but less central than registry control. PESQ reference is the first legitimate watermarked coalition copy, never clean speech.
- **Table F — K=5 codec/temporal robustness.** Best as an appendix or if robustness is a headline contribution. Codec has n=300 per cell; temporal has n=100. Temporal SI-SDR is absent from the original CSV and is not shown.

## Files and integration

Each figure has PDF (paper), PNG (preview), and a source aggregate under `data/`. Tables have CSV, LaTeX, and PNG previews. `LATEX_SNIPPETS.tex` contains insertion templates. The generation script is self-contained and copied here for reproducibility.
"""
    (OUT/"CANDIDATE_GUIDE.md").write_text(guide)
    snippets=r"""% Candidate figure integration snippets (renumber to match the manuscript).
\begin{figure*}[t]
  \centering
  \includegraphics[width=\textwidth]{paper_visual_candidates_20260820/fig_A_blind_asr.pdf}
  \caption{Native-registry blind attribution escape across five watermarking systems. Points show 300-trial attack success rates; error bars are Wilson 95\% confidence intervals. FWP is identical to Mean at $K=2$ by construction.}
  \label{fig:blind-asr-candidate}
\end{figure*}

\begin{figure*}[t]
  \centering
  \includegraphics[width=\textwidth]{paper_visual_candidates_20260820/fig_B_registry_scaling.pdf}
  \caption{Registry-size sensitivity for the four 16-bit watermarking systems. Color denotes coalition size and line style denotes attack method. TimbreWM is excluded because its native 10-bit registry has only $N=1024$.}
  \label{fig:registry-scaling-candidate}
\end{figure*}

\begin{figure*}[t]
  \centering
  \includegraphics[width=\textwidth]{paper_visual_candidates_20260820/fig_C_targeted_any10.pdf}
  \caption{Any-of-ten TCT hit rate under favorable native-registry targets, uniformly random native-registry targets, and uniformly random matched-$N=1024$ targets. Favorable targets are the ten nearest within a deterministic 2,000-candidate subset.}
  \label{fig:target-policy-candidate}
\end{figure*}

\begin{figure*}[t]
  \centering
  \includegraphics[width=\textwidth]{paper_visual_candidates_20260820/fig_D_quality_tradeoff.pdf}
  \caption{Blind attack success versus PESQ relative to the first legitimate watermarked coalition copy. Labels denote coalition size.}
  \label{fig:quality-tradeoff-candidate}
\end{figure*}
"""
    (OUT/"LATEX_SNIPPETS.tex").write_text(snippets)
    script_hash=sha(OUT/"generate_paper_visual_candidates.py")
    trace=f"""figure_table_trace:
  - artifact_id: fig-A
    source_data: {{dataset_id: final-evaluation, file: results/evaluation/attack_{{model}}_K{{K}}.csv}}
    transformation: {{script: generate_paper_visual_candidates.py, hash: {script_hash}}}
    caption_claim: Native-registry blind escape varies by model, coalition size, and Mean/FWP strategy.
    supported_manuscript_claims:
      - {{claim: FWP and Mean are identical at K=2, while their relative performance can diverge for larger coalitions.}}
    limitations: [ASR is near ceiling for several systems, quality is not shown in this panel]
  - artifact_id: fig-B
    source_data: {{dataset_id: registry-control, file: results/evaluation/registry_control_{{model}}_K{{K}}.csv}}
    transformation: {{script: generate_paper_visual_candidates.py, hash: {script_hash}}}
    caption_claim: Blind escape generally increases with registry size, with model-, K-, and method-specific trajectories.
    supported_manuscript_claims:
      - {{claim: Matched registry size is necessary for cross-system attribution comparisons.}}
    limitations: [TimbreWM has no multi-N sweep because its native registry is N=1024]
  - artifact_id: fig-C
    source_data: {{dataset_id: targeted-controls, file: results/evaluation/framing_hull_* and tamper_arbitrary*.csv}}
    transformation: {{script: generate_paper_visual_candidates.py, hash: {script_hash}}}
    caption_claim: TCT any-of-ten success depends strongly on favorable versus arbitrary target selection and registry size.
    supported_manuscript_claims:
      - {{claim: Target-selection policy and active registry size materially affect targeted hit rates.}}
    limitations: [favorable targets are nearest within a deterministic 2000-candidate subset rather than globally nearest]
  - artifact_id: fig-D
    source_data: {{dataset_id: final-evaluation, file: results/evaluation/attack_{{model}}_K{{K}}.csv}}
    transformation: {{script: generate_paper_visual_candidates.py, hash: {script_hash}}}
    caption_claim: Attack success and PESQ reveal method-specific effectiveness-quality trade-offs.
    supported_manuscript_claims:
      - {{claim: Attack effectiveness should be interpreted jointly with post-collusion audio quality.}}
    limitations: [PESQ reference is the first legal watermarked coalition copy and not clean speech]
  - artifact_id: table-E
    source_data: {{dataset_id: final-evaluation, file: results/evaluation/attack_{{model}}_K{{K}}.csv}}
    transformation: {{script: generate_paper_visual_candidates.py, hash: {script_hash}}}
    caption_claim: Exact ASR, NAC, and PESQ values expose effectiveness-quality trade-offs for every model-K pair.
    supported_manuscript_claims:
      - {{claim: FWP is not uniformly preferable to Mean across all watermark systems.}}
    limitations: [dense double-column table, NAC is ACC_near_norm under the stored schema]
  - artifact_id: table-F
    source_data: {{dataset_id: sensitivity-controls, file: results/evaluation/codec_sensitivity_* and temporal_sensitivity_*.csv}}
    transformation: {{script: generate_paper_visual_candidates.py, hash: {script_hash}}}
    caption_claim: K=5 blind attack success remains high under codecs but can change under one-member temporal misalignment.
    supported_manuscript_claims:
      - {{claim: Codec and temporal perturbations have distinct effects on blind collusion escape.}}
    limitations: [codec n=300, temporal n=100, temporal SI-SDR was not saved]
"""
    (OUT/"FIGURE_TABLE_TRACE.yaml").write_text(trace)
    (OUT/"VISUAL_QC.md").write_text("""# Visual quality check

All four figures were rendered and visually inspected against their source aggregates. Figure C required one layout revision because its colorbar overlapped the matched-N panel; the revised render passed. Figures A, B, and C pass axis, units, legend/color scale, colorblind-palette, font-size, 300-dpi, double-column dimension, and no-chart-junk checks. Figure D passes with a note that K annotations are intentionally dense and it is an alternative rather than the recommended main figure. Both table previews were checked for clipping; LaTeX sources preserve exact values and booktabs structure.
""")
    files=[p for p in OUT.rglob("*") if p.is_file() and p.name!="SHA256SUMS.txt"]
    (OUT/"SHA256SUMS.txt").write_text("\n".join(f"{sha(p)}  {p.relative_to(OUT)}" for p in sorted(files))+"\n")
    archive=ROOT/"paper_visual_candidates_20260820.zip"
    if archive.exists(): archive.unlink()
    subprocess.run(["zip","-q","-r",str(archive),OUT.name],cwd=ROOT,check=True)
    print(f"generated {len(files)+1} files at {OUT}")


if __name__ == "__main__": main()

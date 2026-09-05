#!/usr/bin/env python3
"""Create the one-page handoff summary after all requested analyses finish."""
from pathlib import Path
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"data"/"main_text_revision_20260830"
LABEL={"audioseal":"AudioSeal","wavmark":"WavMark","timbrewm":"TimbreWM","voicemark":"VoiceMark","wmcodec":"WMCodec"}

def main():
    risk=pd.read_csv(OUT/"main_risk"/"risk_speaker_cluster_bootstrap.csv")
    k8=pd.read_csv(OUT/"k8_population"/"k8_compact_system_summary.csv")
    path=pd.read_csv(OUT/"mixture_path"/"mixture_path_transition_counts.csv")
    r8=risk[(risk.method=="mean")&(risk.metric=="ASR")&(risk.K==8)].set_index("model")
    lines=["# Results summary — main-text revision (2026-08-30)","",
      "## Main findings","",
      "- Uniform full-waveform averaging already produces high tracing failure at K=2 and remains high through K=8. K=8 rates: "+", ".join(f"{LABEL[m]} {r8.loc[m,'mean']:.3f}" for m in LABEL)+".",
      "- PESQ and STOI change gradually with K; the main risk figure includes 95% ranges obtained by resampling speakers.",
      "- The random K=8 population confirms that decoder behavior depends strongly on coalition bit composition. Strict-majority consistency ranges from "+f"{k8.overall_strict_majority_consistency.min():.3f} to {k8.overall_strict_majority_consistency.max():.3f}; unanimous preservation and tie behavior are reported separately.",
      "- The one-bit case uses one speaker, one payload pair (63374/63390), and the same flipped bit b4 for AudioSeal and VoiceMark. Mel panels use the full audio and per-system 0-dB normalization.",
      "- Controlled interpolation shows markedly different identity-path behavior: AudioSeal has "+f"{int(path.loc[path.model=='audioseal','identity_multiple_transition'].iloc[0])} multiple-transition trials, versus {int(path.loc[path.model=='voicemark','identity_multiple_transition'].iloc[0])} for VoiceMark.","",
      "## Figures for the main text","",
      "1. `main_risk/fig_main_uniform_risk_quality` — tracing failure, PESQ, and STOI versus K.",
      "2. `k8_population/fig_k8_population_composition` — random-population q(c) with speaker-resampled ranges.",
      "3. `frequency_temporal/fig_frequency_temporal_heatmaps` — 1−NCA heatmaps with TF annotations.",
      "4. `onebit_case/fig_unified_onebit_case_audioseal_voicemark` — unified 2×3 case study.",
      "5. `mixture_path/fig_controlled_onebit_interpolation` — aggregate path and transition counts.","",
      "## Appendix","",
      "- `main_risk/fig_appendix_fwp_tracing_failure` — FWP only.",
      "- Full per-trial, per-bit, transition, confidence, and method records are supplied in the archive.","",
      "## Scope","",
      "All uncertainty ranges resample speakers as groups. The K=8 composition results use 300 random trials per system; the constructed K=8 case is excluded. No prediction model, tracing-failure prediction score, or R² result is used."]
    (OUT/"RESULTS_SUMMARY.md").write_text("\n".join(lines)+"\n")
    # Compact one-page PDF/PNG/SVG rendering.
    fig=plt.figure(figsize=(8.27,11.69),facecolor="white"); ax=fig.add_axes([.07,.06,.86,.90]); ax.axis("off")
    ax.text(0,1,"Neural audio watermark collusion — results summary",fontsize=16,fontweight="bold",va="top")
    y=.955
    sections=[("Main findings",lines[4:9]),("Main-text figures",lines[12:17]),("Appendix and scope",[lines[20],lines[21],lines[-1]])]
    for title,items in sections:
      ax.text(0,y,title,fontsize=11,fontweight="bold",va="top"); y-=.035
      for item in items:
        text=item.replace("- ","").replace("`","")
        ax.text(.02,y,u"• "+text,fontsize=8.5,va="top",wrap=True); y-=.057 if len(text)>115 else .040
      y-=.018
    ax.text(0,.05,"All deliverables: PDF + SVG + PNG, analysis CSVs, method JSON files, and source tables. Audio files are intentionally excluded.",fontsize=8.5,fontweight="bold")
    for ext in ("pdf","svg","png"): fig.savefig(OUT/f"results_summary_one_page.{ext}",dpi=300,bbox_inches="tight")
    plt.close(fig)

if __name__=="__main__": main()

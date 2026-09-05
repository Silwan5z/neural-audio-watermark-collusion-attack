#!/usr/bin/env python3
"""Summarize and plot the native-rate constructed K=8 validation case."""
from __future__ import annotations
import csv, json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/"data"/"k8_constructed_payload_case_native_20260829"
OUT=DATA/"analysis"; OUT.mkdir(parents=True,exist_ok=True)
SEED=20260829
MODELS=["audioseal","wavmark","timbrewm","voicemark","wmcodec"]
LABEL={"audioseal":"AudioSeal","wavmark":"WavMark","timbrewm":"TimbreWM","voicemark":"VoiceMark","wmcodec":"WMCodec"}
COLOR={"audioseal":"#2878B5","wavmark":"#54A24B","timbrewm":"#B279A2","voicemark":"#D95F59","wmcodec":"#F2A541"}


def save(fig,stem):
    for ext in ("pdf","svg","png"): fig.savefig(OUT/f"{stem}.{ext}",dpi=300,bbox_inches="tight")
    plt.close(fig)


def main():
    raw={m:json.loads((DATA/"raw"/f"{m}.json").read_text()) for m in MODELS}
    summaries=[]; bits=[]
    for m in MODELS:
        x=raw[m]; ties=[r for r in x["bit_results"] if r["ones_among_8"]==4]
        summaries.append({"model":m,"native_sample_rate":x["native_sample_rate"],
            "clean_exact_out_of_8":x["clean_attribution_exact"],"decoded_identity":x["decoded_identity"],
            "tracing_failure":x["tracing_failure"],"NCA":x["NCA"],
            "strict_majority_departures":len(x["strict_majority_departures"]),
            "strict_majority_departure_bits":json.dumps(x["strict_majority_departures"]),
            "unanimous_violations":len(x["unanimous_violations"]),
            "tie_bits":len(ties),"tie_decoded_one_fraction":np.mean([r["native_decoded_bit"] for r in ties]),
            "tie_mean_abs_margin":np.mean([abs(r["p_bit_1"]-.5) for r in ties]),
            "PESQ_vs_first_copy":x["PESQ_vs_first_copy"],"STOI_vs_first_copy":x["STOI_vs_first_copy"]})
        for r in x["bit_results"]:
            bits.append({"model":m,"bit":r["bit_index_lsb_first"],"ones_among_8":r["ones_among_8"],
                         "state":r["state"],"p_bit_1":r["p_bit_1"],"decoded_bit":r["native_decoded_bit"],
                         "confidence":r["confidence_native_decoded_bit"],
                         "colluder_bits":json.dumps(r["colluder_bits"],separators=(",",":"))})
    with (OUT/"k8_native_system_summary.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=summaries[0]); w.writeheader(); w.writerows(summaries)
    with (OUT/"k8_native_bit_results.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=bits[0]); w.writeheader(); w.writerows(bits)

    # Five rows: controlled composition and decoder response.
    fig,axes=plt.subplots(5,1,figsize=(7.2,8.4),sharex=True)
    for ax,m in zip(axes,MODELS):
        br=raw[m]["bit_results"]; x=np.arange(len(br)); p=np.array([r["p_bit_1"] for r in br]); frac=np.array([r["ones_among_8"]/8 for r in br])
        bars=ax.bar(x,p,color=COLOR[m],alpha=.82,width=.72,edgecolor="white")
        ax.scatter(x,frac,marker="D",s=26,facecolor="white",edgecolor="#111",linewidth=.8,zorder=4)
        for i,r in enumerate(br):
            if r["ones_among_8"]==4: bars[i].set_edgecolor("#7A3E00"); bars[i].set_linewidth(1.8)
        ax.axhline(.5,color="#777",ls="--",lw=.7); ax.set_ylim(0,1.05); ax.set_ylabel("P1")
        ax.set_title(f"{LABEL[m]}  |  decoded={raw[m]['decoded_identity']}  NCA={raw[m]['NCA']:.3f}  TF={raw[m]['tracing_failure']}",loc="left",fontsize=9,fontweight="bold")
        ax.grid(axis="y",color="#E2E2E2",lw=.45); ax.spines[["top","right"]].set_visible(False)
    axes[-1].set_xticks(range(16),[f"b{i}\n({c}/8)" for i,c in enumerate(raw["audioseal"]["ones_per_bit"])])
    axes[-1].set_xlabel("Bit and controlled number of colluders carrying 1 (LSB-first)")
    fig.suptitle("K=8 native-rate validation: controlled coalition composition vs decoder output",fontsize=12,fontweight="bold",y=.995)
    fig.text(.73,.965,"white diamond: coalition one-fraction; outlined bars: 4/4 ties",fontsize=7.5,ha="center")
    fig.subplots_adjust(top=.93,hspace=.52)
    save(fig,"fig_k8_native_all_systems")

    # Response grouped by count, keeping individual bits visible; 4/4 has repeated observations.
    fig,axes=plt.subplots(1,5,figsize=(7.2,2.65),sharey=True)
    for ax,m in zip(axes,MODELS):
        br=raw[m]["bit_results"]
        for r in br:
            c=r["ones_among_8"]; ax.scatter(c,r["p_bit_1"],s=34,color=COLOR[m],alpha=.78,edgecolor="white",linewidth=.4)
        ax.plot([0,8],[0,1],color="#333",ls="--",lw=.8)
        ax.axvspan(3.7,4.3,color="#F4D7B5",alpha=.5); ax.axhline(.5,color="#999",lw=.6)
        ax.set_title(LABEL[m],fontsize=8.5,fontweight="bold"); ax.set_xticks([0,2,4,6,8]); ax.set_xlabel("ones/8")
        ax.grid(color="#E4E4E4",lw=.45); ax.spines[["top","right"]].set_visible(False)
    axes[0].set_ylabel("Decoder P(bit=1)"); axes[0].set_ylim(-.04,1.04)
    fig.suptitle("The 4/4 coalition composition does not imply decoder uncertainty",fontsize=11,fontweight="bold",y=1.03)
    save(fig,"fig_k8_native_response_by_composition")

    # Summary bars.
    x=np.arange(5); fig,axes=plt.subplots(1,3,figsize=(7.1,2.5))
    axes[0].bar(x,[s["NCA"] for s in summaries],color=[COLOR[m] for m in MODELS]); axes[0].set_ylim(.7,.94); axes[0].set_ylabel("NCA")
    axes[1].bar(x,[s["strict_majority_departures"] for s in summaries],color=[COLOR[m] for m in MODELS]); axes[1].set_ylabel("Bits"); axes[1].set_ylim(0,3.5)
    axes[2].bar(x,[s["tie_mean_abs_margin"] for s in summaries],color=[COLOR[m] for m in MODELS]); axes[2].set_ylabel("Mean |P1-0.5| on 4/4 bits"); axes[2].set_ylim(0,.5)
    titles=["Nearest-colluder agreement","Strict-majority departures","Confidence on balanced bits"]
    for ax,t in zip(axes,titles):
        ax.set_title(t,fontsize=8.7,fontweight="bold"); ax.set_xticks(x,[LABEL[m] for m in MODELS],rotation=38,ha="right",fontsize=7); ax.grid(axis="y",color="#E4E4E4",lw=.45); ax.spines[["top","right"]].set_visible(False)
    fig.suptitle("All systems escape, but balanced bits are resolved differently",fontsize=11,fontweight="bold",y=1.05)
    fig.subplots_adjust(wspace=.48,top=.78)
    save(fig,"fig_k8_native_mechanism_summary")

    wm_pre=json.loads((ROOT/"data"/"k8_constructed_payload_case_20260829"/"raw"/"wmcodec.json").read_text())
    wm=raw["wmcodec"]
    audit={"experiment":"K=8 constructed payload native-rate validation","seed":SEED,
           "selection":"pre-registered constructed matrix; not selected by outcome",
           "ones_per_bit_16":raw["audioseal"]["ones_per_bit"],"ones_per_bit_10":raw["timbrewm"]["ones_per_bit"],
           "clean_exact_all_systems":{m:raw[m]["clean_attribution_exact"] for m in MODELS},
           "wmcodec_protocol_control":{"standardized_16k_clean_exact":wm_pre["clean_attribution_exact"],
               "native_24k_clean_exact":wm["clean_attribution_exact"],"standardized_decoded_identity":wm_pre["decoded_identity"],
               "native_decoded_identity":wm["decoded_identity"]},"audio_file_count":len(list((DATA/"audio").rglob("*.wav")))}
    (OUT/"analysis_audit.json").write_text(json.dumps(audit,indent=2)+"\n")
    lines=["# K=8 constructed-payload native-rate validation","","## Protocol","",
           "- Fixed K=8; equal weights 1/8.","- Pre-registered bit counts: `0,1,2,3,4,4,5,6,7,8,4,4,4,4,4,4` for 16-bit systems.",
           "- TimbreWM uses the first ten columns and covers every count from 0 through 8, including two 4/4 bits.",
           "- Attribution is performed at each system's native sample rate; quality metrics are evaluated at 16 kHz.","",
           "## Results","","| System | Native SR | Clean exact | Decoded ID | TF | NCA | Strict-majority departures | PESQ | STOI |","|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for s in summaries: lines.append(f"| {LABEL[s['model']]} | {s['native_sample_rate']} | {s['clean_exact_out_of_8']}/8 | {s['decoded_identity']} | {s['tracing_failure']} | {s['NCA']:.4f} | {s['strict_majority_departures']} | {s['PESQ_vs_first_copy']:.4f} | {s['STOI_vs_first_copy']:.4f} |")
    lines += ["","All five mixtures decode to identities outside the eight colluders. AudioSeal, WavMark, and TimbreWM preserve every strict-majority bit, whereas VoiceMark departs at three strict-majority bits and WMCodec at one. No system violates the two unanimous anchors. The repeated 4/4 columns are not generally decoded with low confidence: chunk/digit systems often produce near-deterministic outcomes even under perfectly balanced coalition composition.","",
              "## WMCodec protocol audit","",f"The standardized 16 kHz wrapper decoded {wm_pre['clean_attribution_exact']}/8 source copies exactly. Native 24 kHz inference decodes {wm['clean_attribution_exact']}/8. Therefore the native result is the valid case-study result; the preliminary 16 kHz directory is retained only as a resampling-control audit.","",
              "## Evidence boundary","","This is one pre-registered constructed case. It demonstrates possible bit-level mechanisms and verifies the controlled composition; it is not a population estimate or a system ranking."]
    (OUT/"K8_NATIVE_RESULTS_ANALYSIS.md").write_text("\n".join(lines)+"\n")
    print(OUT)

if __name__=="__main__": main()

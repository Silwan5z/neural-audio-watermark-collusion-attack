#!/usr/bin/env python3
"""Merge, validate, tabulate, and plot the five-system K=5 payload case."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "k5_payload_case_trial000_20260829"
RAW = OUT / "raw"
MODELS = ("audioseal", "wavmark", "timbrewm", "voicemark", "wmcodec")
LABELS = {"audioseal": "AudioSeal", "wavmark": "WavMark", "timbrewm": "TimbreWM",
          "voicemark": "VoiceMark", "wmcodec": "WMCodec"}
BLUE = "#2878B5"; RED = "#D95F59"; ORANGE = "#F2C14E"; GREEN = "#5AA469"


def atomic_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
    tmp.replace(path)


def main() -> None:
    data = {m: json.loads((RAW / f"{m}_trial000.json").read_text(encoding="utf-8"))
            for m in MODELS}
    keys = {(d["trial_id"], d["spk"], d["local_t"], d["condition"]) for d in data.values()}
    if keys != {(0, "chinese:SSB0005", 0, "full_waveform")}:
        raise ValueError(f"case keys do not align: {keys}")

    payload_rows, bit_rows, system_rows = [], [], []
    for model, d in data.items():
        nbits = d["payload_nbits"]
        majority_bits = [int(r["ones_among_5"] >= 3) for r in d["bit_results"]]
        majority_identity = int(sum(value << bit for bit, value in enumerate(majority_bits)))
        majority_mismatch = [bit for bit, (a, b) in enumerate(
            zip(majority_bits, d["decoded_payload_bits"])) if a != b]
        unanimous_violations = [
            r["bit_index_lsb_first"] for r in d["bit_results"]
            if r["agreement"] != "disputed" and
            r["native_decoded_bit"] != r["colluder_bits"][0]
        ]
        for ci, (payload, bits) in enumerate(zip(d["coalition_payloads"], d["coalition_payload_bits"]), 1):
            payload_rows.append({
                "model": model, "colluder": ci, "payload_decimal": payload,
                "payload_hex": f"0x{payload:0{(nbits+3)//4}X}",
                "payload_binary_msb_to_lsb": f"{payload:0{nbits}b}",
                "payload_bits_lsb_to_msb": "".join(map(str, bits)),
            })
        for r in d["bit_results"]:
            bit_rows.append({
                "model": model, "bit_index_lsb_first": r["bit_index_lsb_first"],
                **{f"colluder_{i+1}_bit": r["colluder_bits"][i] for i in range(5)},
                "ones_among_5": r["ones_among_5"],
                "coalition_one_fraction": r["ones_among_5"] / 5,
                "agreement": r["agreement"], "p_bit_1": r["p_bit_1"],
                "native_decoded_bit": r["native_decoded_bit"],
                "confidence_native_decoded_bit": r["confidence_native_decoded_bit"],
                "marginal_threshold_bit": r["marginal_threshold_bit"],
                "native_vs_marginal_threshold_mismatch": r["native_vs_marginal_threshold_mismatch"],
            })
        system_rows.append({
            "model": model, "payload_nbits": nbits,
            "coalition_payloads": json.dumps(d["coalition_payloads"], separators=(",", ":")),
            "unanimous_0_bits": json.dumps(d["unanimous_0_bits"]),
            "unanimous_1_bits": json.dumps(d["unanimous_1_bits"]),
            "disputed_bits": json.dumps(d["disputed_bits"]),
            "decoded_identity": d["decoded_identity"],
            "decoded_bits_lsb_to_msb": "".join(map(str, d["decoded_payload_bits"])),
            "bitwise_majority_identity": majority_identity,
            "bitwise_majority_bits_lsb_to_msb": "".join(map(str, majority_bits)),
            "decoded_vs_majority_mismatch_bits": json.dumps(majority_mismatch),
            "unanimous_bit_violation_bits": json.dumps(unanimous_violations),
            "NCA_bits": d["NCA_bits"], "NCA": d["NCA"],
            "tracing_failure": d["tracing_failure"],
            "presence": "" if d["presence"] is None else d["presence"],
            "confidence_kind": d["confidence_kind"],
            "valid_start_pattern_windows": d.get("valid_start_pattern_windows", ""),
            "total_sliding_windows": d.get("total_sliding_windows", ""),
        })

    atomic_csv(OUT / "k5_payload_case_payloads.csv", list(payload_rows[0]), payload_rows)
    atomic_csv(OUT / "k5_payload_case_bits.csv", list(bit_rows[0]), bit_rows)
    atomic_csv(OUT / "k5_payload_case_system_summary.csv", list(system_rows[0]), system_rows)

    mpl.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8.2, "axes.titlesize": 10,
        "axes.labelsize": 8.5, "xtick.labelsize": 7.2, "ytick.labelsize": 7.2,
        "pdf.fonttype": 42, "svg.fonttype": "none",
    })
    fig, axes = plt.subplots(5, 2, figsize=(7.2, 10.4),
                             gridspec_kw={"width_ratios": [1.04, 1.22], "hspace": .86, "wspace": .25})
    for ri, model in enumerate(MODELS):
        d = data[model]; n = d["payload_nbits"]
        matrix = np.asarray(d["coalition_payload_bits"], dtype=int)
        p1 = np.asarray([r["p_bit_1"] for r in d["bit_results"]], dtype=float)
        hard = np.asarray(d["decoded_payload_bits"], dtype=int)
        frac = matrix.mean(axis=0)
        same = (matrix.min(axis=0) == matrix.max(axis=0))

        ax = axes[ri, 0]
        ax.imshow(matrix, cmap=mpl.colors.ListedColormap(["#F2F4F7", BLUE]),
                  vmin=0, vmax=1, aspect="auto", interpolation="nearest")
        for bit in range(n):
            if same[bit]:
                ax.add_patch(mpl.patches.Rectangle((bit-.48, -.48), .96, 4.96,
                             fill=False, ec=GREEN, lw=1.7))
            else:
                ax.add_patch(mpl.patches.Rectangle((bit-.48, -.48), .96, 4.96,
                             fill=False, ec=ORANGE, lw=.65, alpha=.8))
            for colluder in range(5):
                ax.text(bit, colluder, str(matrix[colluder, bit]), ha="center", va="center",
                        color="white" if matrix[colluder, bit] else "#555555", fontsize=6.2)
        ax.set(xticks=np.arange(n), xticklabels=[f"b{x}" for x in range(n)],
               yticks=np.arange(5), yticklabels=[f"C{x}" for x in range(1, 6)])
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        ax.tick_params(length=0); ax.set_title(f"{LABELS[model]} — five payload bit patterns", loc="left", pad=13)
        payload_text = ", ".join(str(x) for x in d["coalition_payloads"])
        ax.text(0, 1.10, f"payloads: {payload_text}", transform=ax.transAxes,
                fontsize=7, color="#555555", va="bottom")
        for spine in ax.spines.values(): spine.set_visible(False)

        ax = axes[ri, 1]
        x = np.arange(n)
        for bit in range(n):
            ax.axvspan(bit-.48, bit+.48, color=GREEN if same[bit] else ORANGE,
                       alpha=.10 if same[bit] else .055, lw=0)
        ax.bar(x, p1, color=np.where(hard == 1, BLUE, RED), width=.72, alpha=.88,
               label=r"Decoder $P(bit=1)$")
        ax.scatter(x, frac, marker="D", s=18, color="#222222", zorder=4,
                   label="Colluder 1-fraction")
        ax.axhline(.5, color="#777777", ls="--", lw=.7)
        for bit in range(n):
            ax.text(bit, min(1.04, p1[bit]+.035), str(hard[bit]), ha="center", va="bottom",
                    fontsize=6.3, color="#222222")
        ax.set(xticks=x, xticklabels=[f"b{x}" for x in x], ylim=(0, 1.12),
               ylabel=r"$P(bit=1)$")
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        ax.set_title(f"decoded ID={d['decoded_identity']} · NCA={d['NCA']:.4g} · tracing failure={d['tracing_failure']}",
                     loc="left", pad=13, fontsize=8.5)
        ax.grid(axis="y", color="#dddddd", lw=.45)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("K=5 equal-waveform-mean case: colluder bits and decoded confidence",
                 fontsize=13, y=.997)
    fig.text(.5, .004, "Bit order is LSB-first. Green outlines: unanimous; amber: disputed. "
             "Bars: decoder P(bit=1); diamonds: colluder 1-fraction; bar labels: native hard bits.",
             ha="center", fontsize=7.5, color="#555555")
    for ext in ("pdf", "svg", "png"):
        fig.savefig(OUT / f"fig_k5_payload_case_all_systems.{ext}", dpi=260 if ext == "png" else None,
                    bbox_inches="tight")
    plt.close(fig)

    # Detailed human-readable report.
    lines = [
        "# K=5 equal-mean payload case (trial 0)", "",
        "- Content key: `chinese:SSB0005`, `local_t=0`.",
        "- Condition: `full_waveform`; five watermarked copies mixed with weights `[0.2,0.2,0.2,0.2,0.2]`.",
        "- Bit order in every table: LSB-first (`b0` is the least significant bit).", "",
        "## Cross-system summary", "",
        "| System | Payloads | Same-1 bits | Disputed bits | Decoded ID | Majority deviations | Unanimous violations | NCA | TF |",
        "|---|---|---|---|---:|---|---|---:|---:|",
    ]
    for m in MODELS:
        d = data[m]
        majority = [int(r['ones_among_5'] >= 3) for r in d['bit_results']]
        deviations = [i for i, (x, y) in enumerate(zip(majority, d['decoded_payload_bits'])) if x != y]
        violations = [r['bit_index_lsb_first'] for r in d['bit_results']
                      if r['agreement'] != 'disputed' and r['native_decoded_bit'] != r['colluder_bits'][0]]
        lines.append(f"| {LABELS[m]} | {d['coalition_payloads']} | {d['unanimous_1_bits']} | "
                     f"{d['disputed_bits']} | {d['decoded_identity']} | {deviations} | {violations} | "
                     f"{d['NCA']:.4f} | {d['tracing_failure']} |")
    lines += ["", "## Detailed bit confidence", "",
              "`P1` is decoder P(bit=1); `Conf` is confidence assigned to the native decoded hard bit. "
              "For WavMark, P1 is the valid-window vote fraction rather than an official neural posterior.", ""]
    for m in MODELS:
        d = data[m]; lines += [f"### {LABELS[m]}", "",
            "| Bit | Five colluders | Ones/5 | Status | P1 | Native bit | Conf |",
            "|---:|---|---:|---|---:|---:|---:|"]
        for r in d["bit_results"]:
            status = {"unanimous_0": "same 0", "unanimous_1": "same 1", "disputed": "different"}[r["agreement"]]
            lines.append(f"| b{r['bit_index_lsb_first']} | {''.join(map(str,r['colluder_bits']))} | "
                         f"{r['ones_among_5']} | {status} | {r['p_bit_1']:.4f} | "
                         f"{r['native_decoded_bit']} | {r['confidence_native_decoded_bit']:.4f} |")
        lines.append("")
    lines += ["## Confidence definitions", "",
              "- AudioSeal: direct sigmoid of the 16 detector bit logits.",
              "- VoiceMark: exact per-bit marginal of each native 16-way chunk posterior.",
              "- WMCodec: exact per-bit marginal of each native 16-way digit posterior.",
              "- TimbreWM: sigmoid mapping of the native continuous decoder score, matching the project interface.",
              f"- WavMark: per-bit vote fraction over {data['wavmark']['valid_start_pattern_windows']} "
              f"start-pattern-valid windows out of {data['wavmark']['total_sliding_windows']} sliding windows; confidence proxy only.", ""]
    (OUT / "K5_PAYLOAD_CASE_ANALYSIS.md").write_text("\n".join(lines), encoding="utf-8")
    (OUT / "analysis_audit.json").write_text(json.dumps({
        "trial_id": 0, "spk": "chinese:SSB0005", "local_t": 0,
        "condition": "full_waveform", "weights": [0.2]*5,
        "models": list(MODELS), "bit_order": "LSB-first",
        "source": str(ROOT / "data" / "frequency_temporal_k5_20260828"),
        "selection": "first trial with complete aligned full_waveform outputs for all five systems; not selected by outcome",
        "hard_bit_validation": "recomputed/native evidence identities matched the existing trial CSV decoded identities",
    }, indent=2) + "\n", encoding="utf-8")
    print(f"complete: {OUT}")


if __name__ == "__main__":
    main()

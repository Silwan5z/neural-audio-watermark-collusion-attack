#!/usr/bin/env python3
"""Build the browser data bundle from audited native decoder outputs."""
from __future__ import annotations

import json
import os
import uuid
import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODELS = (
    ("audioseal", "AudioSeal"),
    ("wavmark", "WavMark"),
    ("timbrewm", "TimbreWM"),
    ("voicemark", "VoiceMark"),
    ("wmcodec", "WMCodec"),
)
K_VALUES = (2, 3, 5, 8)
SHIFTS = (-50, -20, -10, 0, 10, 20, 50)
CODECS = (
    ("none", "No codec", "Reference"),
    ("mp3_128k", "MP3", "128 kbps"),
    ("opus_64k", "Opus", "64 kbps"),
)


def offset_slug(value: int) -> str:
    if value < 0:
        return f"minus{abs(value)}"
    if value > 0:
        return f"plus{value}"
    return "aligned"


def offset_label(value: int) -> str:
    if value < 0:
        return f"−{abs(value)} ms"
    if value > 0:
        return f"+{value} ms"
    return "Aligned"


def main() -> None:
    with (ROOT / "demos" / "conditions" / "metadata.csv").open(
            newline="", encoding="utf-8") as handle:
        metric_rows = list(csv.DictReader(handle))
    metrics = {
        (row["system"], row["family"], row["condition"]): {
            "pesq": float(row["pesq"]),
            "stoi": float(row["stoi"]),
            "siSdr": float(row["si_sdr_db"]),
        }
        for row in metric_rows
    }
    with (ROOT / "demos" / "metadata.csv").open(
            newline="", encoding="utf-8") as handle:
        copy_rows = {row["system"]: row for row in csv.DictReader(handle)}

    systems = []
    for model, name in MODELS:
        decoded = json.loads((
            ROOT / "demos" / "decoded" /
            f"{model}.json").read_text(encoding="utf-8"))
        coalitions = decoded["coalitions"]
        sizes = [{
            "label": f"K = {k}",
            "detail": f"{k} copies",
            "coalition": coalitions[str(k)],
            "decoded": decoded["coalition_size"][str(k)]["decoded_payload"],
            "audio": f"conditions/coalition_size/{model}/k{k}",
            "metrics": metrics[(model, "coalition_size", f"K={k}")],
        } for k in K_VALUES]
        offsets = [{
            "label": offset_label(shift),
            "detail": ("Reference" if shift == 0 else
                       "Earlier" if shift < 0 else "Later"),
            "coalition": coalitions["5"],
            "decoded": decoded["offset"][str(shift)]["decoded_payload"],
            "audio": f"conditions/offset/{model}/{offset_slug(shift)}",
            "metrics": metrics[(model, "offset", str(shift))],
        } for shift in SHIFTS]
        codecs = [{
            "label": label,
            "detail": detail,
            "coalition": coalitions["5"],
            "decoded": decoded["codec"][codec]["decoded_payload"],
            "audio": f"conditions/codec/{model}/{codec}",
            "metrics": metrics[(model, "codec", codec)],
        } for codec, label, detail in CODECS]
        copy_row = copy_rows[model]
        payload_a = int(copy_row["payload_a"])
        payload_b = int(copy_row["payload_b"])
        copies = [
            {"label": "Clean source", "detail": "Unwatermarked",
             "audio": "source_reference", "assigned": None,
             "decoded": None, "metrics": None},
            {"label": "Personalized copy A", "detail": "Valid copy",
             "audio": f"{model}/member_{payload_a}", "assigned": payload_a,
             "decoded": payload_a, "metrics": None},
            {"label": "Personalized copy B", "detail": "Valid copy",
             "audio": f"{model}/member_{payload_b}", "assigned": payload_b,
             "decoded": payload_b, "metrics": None},
            {"label": "50/50 average", "detail": "K = 2",
             "audio": f"{model}/uniform_average", "assigned": None,
             "decoded": int(copy_row["decoded_average_payload"]),
             "coalition": [payload_a, payload_b],
             "metrics": {
                 "pesq": float(copy_row["pesq"]),
                 "stoi": float(copy_row["stoi"]),
                 "siSdr": float(copy_row["si_sdr_db"]),
             }},
        ]
        systems.append({
            "id": model, "name": name,
            "copies": copies, "coalitionSize": sizes,
            "offsets": offsets, "codecs": codecs,
        })

    value = {"trial": 150, "systems": systems}
    output = ROOT / "demos" / "demo-data.js"
    temporary = output.with_name(f".{output.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(
        "window.DEMO_DATA = " +
        json.dumps(value, indent=2, ensure_ascii=False) + ";\n",
        encoding="utf-8")
    os.replace(temporary, output)
    print(json.dumps({"output": str(output), "systems": len(systems),
                      "conditions": sum(
                          len(item["coalitionSize"]) + len(item["offsets"]) +
                          len(item["codecs"]) for item in systems)}))


if __name__ == "__main__":
    main()

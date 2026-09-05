"""Fail-closed validation for final single-copy attack and tamper outputs."""
from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from registry import (  # noqa: E402
    NBITS, full_registry_bits, full_registry_size, int_to_bits,
)
from registry_size_control import active_registries  # noqa: E402
from run_single_tamper import (  # noqa: E402
    most_reachable_targets, tamper_seed,
)


EVALUATION = ROOT / "results" / "evaluation"
MODELS = ["audioseal", "wavmark", "voicemark", "wmcodec", "timbrewm"]
DIFF_MODELS = ["audioseal", "voicemark", "wmcodec", "timbrewm"]
KS = [2, 3, 5, 8]
QUALITY = ["PESQ", "STOI", "SI_SDR"]
SOFTMIN_REFERENCE = "first_legitimate_watermarked_coalition_copy"
SINGLE_REFERENCE = "source_watermarked_audio_before_tamper"


def read(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def require_finite(row: dict[str, str], fields: list[str], context: str) -> None:
    for field in fields:
        require(row.get(field, "") != "", f"{context}: empty {field}")
        require(np.isfinite(float(row[field])), f"{context}: non-finite {field}")


def registry_set(model: str) -> set[int]:
    return {1024} if model == "timbrewm" else {1024, 65536}


def validate_attack_evasion() -> None:
    stems = []
    for model in MODELS:
        stems.extend([
            f"attack_evasion_overwrite_same_{model}",
            f"attack_evasion_encodec_{model}",
            f"attack_evasion_vocoder_{model}",
        ])
    stems.extend(f"attack_evasion_fgsm_{model}" for model in DIFF_MODELS)
    cross = {
        "audioseal": "wavmark", "wavmark": "voicemark",
        "voicemark": "wmcodec", "wmcodec": "timbrewm",
        "timbrewm": "audioseal",
    }
    stems.extend(
        f"attack_evasion_overwrite_cross_{model}__to_{target}"
        for model, target in cross.items()
    )
    require(len(stems) == 24, "internal attack manifest must have 24 cells")

    for stem in stems:
        rows = read(EVALUATION / f"{stem}.csv")
        model = rows[0]["model"] if rows else ""
        expected_registries = registry_set(model)
        require(len(rows) == 300 * len(expected_registries),
                f"{stem}: wrong row count {len(rows)}")
        grouped: dict[int, list[dict[str, str]]] = defaultdict(list)
        for row in rows:
            grouped[int(row["trial_id"])].append(row)
            require(row["mode"] == "evasion", f"{stem}: wrong mode")
            require_finite(row, QUALITY, stem)
        require(set(grouped) == set(range(300)), f"{stem}: incomplete trials")
        for trial, trial_rows in grouped.items():
            require({int(row["N_registry"]) for row in trial_rows} == expected_registries,
                    f"{stem} trial={trial}: wrong registries")
            require(len({row["source_id"] for row in trial_rows}) == 1,
                    f"{stem} trial={trial}: source changed across registries")
            for field in QUALITY:
                require(len({row[field] for row in trial_rows}) == 1,
                        f"{stem} trial={trial}: {field} differs across registries")
    print("validated attack evasion: 24 cells", flush=True)


def validate_softmin() -> None:
    for model in MODELS:
        for k in KS:
            source_path = EVALUATION / f"margin_reachability_v2_{model}_K{k}.csv"
            source = read(source_path)
            require(len(source) == 3000, f"{source_path.name}: wrong row count")
            grouped: dict[int, list[dict[str, str]]] = defaultdict(list)
            source_map = {}
            for row in source:
                gi, target = int(row["gi"]), int(row["target"])
                grouped[gi].append(row)
                source_map[(gi, target)] = row
                require(int(row["N_registry"]) == 1024,
                        f"{source_path.name}: source registry is not 1024")
                require(row.get("quality_reference") == SOFTMIN_REFERENCE,
                        f"{source_path.name}: wrong quality reference")
                require_finite(row, QUALITY, source_path.name)
                weights = np.asarray(json.loads(row["weights"]), dtype=float)
                require(weights.shape == (k,), f"{source_path.name}: wrong weight shape")
                require(np.isclose(weights.sum(), 1.0, atol=1e-6),
                        f"{source_path.name}: weights do not sum to one")
            require(set(grouped) == set(range(300)),
                    f"{source_path.name}: incomplete trials")
            require(all(len({int(row['target']) for row in rows}) == 10
                        for rows in grouped.values()),
                    f"{source_path.name}: trial does not have 10 unique targets")

            if model == "timbrewm":
                continue
            native_path = EVALUATION / f"margin_reachability_v2_native_{model}_K{k}.csv"
            native = read(native_path)
            require(len(native) == 3000, f"{native_path.name}: wrong row count")
            require(len({(int(row["gi"]), int(row["target"])) for row in native}) == 3000,
                    f"{native_path.name}: duplicate or missing keys")
            for row in native:
                key = (int(row["gi"]), int(row["target"]))
                require(key in source_map, f"{native_path.name}: target absent from N=1024")
                base = source_map[key]
                require(int(row["selection_N_registry"]) == 1024,
                        f"{native_path.name}: wrong selection registry")
                require(int(row["N_registry"]) == 65536,
                        f"{native_path.name}: wrong native registry")
                for field in ["spk", "local_t", "weights", "reachability_score",
                              "effective_K", "solver_success", *QUALITY,
                              "quality_reference"]:
                    require(row[field] == base[field],
                            f"{native_path.name} key={key}: {field} changed vs N=1024")
    print("validated softmin-v2: 20 N=1024 cells + 16 native cells", flush=True)


def validate_single_tamper() -> None:
    manifest = {
        "overwrite": MODELS,
        "ifgsm": DIFF_MODELS,
    }
    for attack, models in manifest.items():
        for model in models:
            path = EVALUATION / f"single_tamper_{attack}_{model}.csv"
            rows = read(path)
            expected_registries = registry_set(model)
            require(len(rows) == 300 * 10 * len(expected_registries),
                    f"{path.name}: wrong row count {len(rows)}")
            grouped: dict[int, list[dict[str, str]]] = defaultdict(list)
            for row in rows:
                grouped[int(row["trial_id"])].append(row)
                require(row["mode"] == "targeted_single_copy_top10_reachable",
                        f"{path.name}: wrong mode")
                require(row.get("quality_reference") == SINGLE_REFERENCE,
                        f"{path.name}: wrong quality reference")
                require_finite(row, [*QUALITY, "SNR"], path.name)
                parameters = json.loads(row["attack_parameters"])
                if attack == "ifgsm":
                    require(int(parameters["steps"]) == 10,
                            f"{path.name}: I-FGSM is not 10-step")
            require(set(grouped) == set(range(300)), f"{path.name}: incomplete trials")

            for trial_id, trial_rows in grouped.items():
                require({int(row["N_registry"]) for row in trial_rows} == expected_registries,
                        f"{path.name} trial={trial_id}: wrong registries")
                by_registry = defaultdict(list)
                for row in trial_rows:
                    by_registry[int(row["N_registry"])].append(row)
                for registry, registry_rows in by_registry.items():
                    require(len(registry_rows) == 10,
                            f"{path.name} trial={trial_id} N={registry}: not 10 rows")
                    require({int(row["candidate_rank"]) for row in registry_rows}
                            == set(range(1, 11)),
                            f"{path.name} trial={trial_id}: bad candidate ranks")
                    require(len({int(row["target_id"]) for row in registry_rows}) == 10,
                            f"{path.name} trial={trial_id}: duplicate targets")
                    hits = sum(int(row["target_hit"]) for row in registry_rows)
                    require(all(int(row["successful_targets_in_top10"]) == hits
                                for row in registry_rows),
                            f"{path.name} trial={trial_id}: wrong top-10 hit count")

                representative = sorted(
                    by_registry[min(expected_registries)],
                    key=lambda row: int(row["candidate_rank"]),
                )
                first = representative[0]
                source_id = int(first["source_id"])
                spk, local_t = first["spk"], int(first["local_t"])
                require(int(first["seed"]) == tamper_seed(spk, model, local_t),
                        f"{path.name} trial={trial_id}: wrong seed")
                selection_size = min(1024, full_registry_size(model))
                require(int(first["selection_N_registry"]) == selection_size,
                        f"{path.name} trial={trial_id}: wrong selection N")
                source_bits = int_to_bits(source_id, NBITS[model])
                registries = active_registries(
                    full_registry_size(model), [source_id],
                    sorted({selection_size, *expected_registries}), spk, 1, local_t)
                candidates = registries[selection_size]
                candidates = candidates[candidates != source_id]
                expected = most_reachable_targets(
                    source_bits, candidates, full_registry_bits(model))
                observed = [(int(row["target_id"]), float(row["reachability_score"]))
                            for row in representative]
                require([target for target, _ in observed]
                        == [target for target, _ in expected],
                        f"{path.name} trial={trial_id}: targets are not top-10 reachable")
                require(np.allclose([score for _, score in observed],
                                    [score for _, score in expected], atol=1e-8),
                        f"{path.name} trial={trial_id}: wrong reachability scores")

                if len(expected_registries) == 2:
                    small = {int(row["candidate_rank"]): row for row in by_registry[1024]}
                    native = {int(row["candidate_rank"]): row for row in by_registry[65536]}
                    for rank in range(1, 11):
                        for field in ["source_id", "target_id", "target_bits", "PESQ",
                                      "STOI", "SI_SDR", "SNR", "presence",
                                      "decoded_bits", "attack_parameters"]:
                            require(small[rank][field] == native[rank][field],
                                    f"{path.name} trial={trial_id} rank={rank}: "
                                    f"{field} changed across registries")
    print("validated single-copy tamper: 9 cells", flush=True)


def main() -> None:
    validate_attack_evasion()
    validate_softmin()
    validate_single_tamper()
    print("FINAL VALIDATION PASSED: publication is safe", flush=True)


if __name__ == "__main__":
    main()

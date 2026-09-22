# Script map

[Back to the overview](../README.md) ·
[Full reproduction guide](../REPRODUCIBILITY.md) ·
[Released-data guide](../data/README.md)

Scripts are grouped by the question they answer. Experiment runners write to
ignored `results/`; only validated, fixed release records belong under `data/`.

## Main evaluation pipeline

| Stage | Entry point | Purpose |
|---|---|---|
| Uniform averaging | `run_average.py` | K=2, 3, and 5 native-rate averaging with valid source copies |
| Uniform merge | `merge_average.py` | Merge shards and require 300 unique trials per model/K |
| K=8 evidence | `run_average_k8.py` | K=8 averaging plus native bit/latent evidence |
| K=8 source audit | `validate_average_k8.py` | Validate or replace invalid source copies |
| Target coalitions | `prepare_coalitions.py` | Build shared valid K=5/K=8 coalitions |
| Targeted attack | `run_targeted.py` | Evaluate ten Target-Bit Margin nonmembers per trial |
| Targeted merge | `merge_targeted.py` | Merge shards and verify 300 complete trials |

`native_audio.py` contains the shared native-rate embedding/decoding path.
`target_bit_margin.py` contains the mixture-weight optimizer.

## Mechanism and defense analyses

| Analysis | Entry point | Output role |
|---|---|---|
| One-bit pairs | `run_one_bit_pairs.py` | Construct valid AudioSeal/VoiceMark payload pairs differing by one bit |
| One-bit paths | `run_one_bit_paths.py` | Sweep mixture weights between each pair |
| One-bit summary | `summarize_one_bit.py` | Produce the compact path records |
| Compact confidence | `collect_confidence.py` | Minimum confidence for Single, Average, and selected Targeted outputs |
| Full confidence | `collect_confidence_full.py` | Complete per-bit vectors for screening |
| Confidence screen | `screen_confidence.py` | Speaker-disjoint five-fold calibration and evaluation |
| Confidence merge | `merge_confidence_screening.py` | Build the five-row paper screening table |
| Confidence export | `export_confidence_screening.py` | Release sufficient statistics, folds, and thresholds for independent verification |

## Additional evaluations

| Analysis | Runner | Summarizer |
|---|---|---|
| SI-SDR and SNR | `compute_uniform_quality.py` | `summarize_uniform_quality.py` |
| ViSQOL | `compute_uniform_visqol.py` | `summarize_uniform_quality.py` |
| Temporal offsets | `run_alignment_stress_test.py` | `summarize_alignment_stress_test.py` |
| MP3 and Opus | `run_codec_stress_test.py` | `summarize_codec_stress_test.py` (including post-codec single-copy controls) |
| Registry occupancy | — | `analyze_registry_occupancy.py` |

The core paper tables can be regenerated without model inference using
`summarize_average.py` and `summarize_targeted.py`. Their default outputs are
under `results/summary/`; tracked release files are never overwritten by
default.

All model inference stays at the backend's native rate: 16 kHz for AudioSeal,
WavMark, and VoiceMark; 22.05 kHz for TimbreWM; and 24 kHz for WMCodec.
Quality metrics receive 16 kHz copies only after native-rate decoding.

## Figures and release tools

| Tool | Purpose |
|---|---|
| `figures/build_composition.py` | Coalition bit-support response figure |
| `figures/build_paths.py` | One-bit mixture-path figure |
| `figures/build_confidence.py` | Minimum-confidence distribution figure |
| `export_demo_audio.py` | Export all five systems across K, timing, and codec demo conditions |
| `decode_demo_audio.py` | Decode all 14 native-rate demo conditions for one system |
| `build_demo_data.py` | Assemble the five-system browser data bundle from decoded records |
| `build_data_manifest.py` | Rebuild sizes, row counts, and SHA-256 hashes |
| `verify_release.py` | Validate the entire public release without model inference |

## Sharding convention

Long runners accept `--shard-id` and `--num-shards`. A typical seven-way launch
uses shard IDs 0 through 6; each shard resumes from its own checkpoint. Merge or
summary scripts should run only after every shard has completed.

The full commands, fixed seeds, optimization settings, data preparation, and
interpretation limits are documented in
[`REPRODUCIBILITY.md`](../REPRODUCIBILITY.md).

# Released data guide

[Back to the overview](../README.md) ·
[Read the result tables](../RESULTS.md) ·
[Reproduce the evaluation](../REPRODUCIBILITY.md)

Only records used by the current paper and its documented additional analyses
are tracked here. Start with a compact CSV if you want a table value; use the
corresponding full-record link when you need trial-level evidence.

## Start with these files

| Result | Compact table | Complete records |
|---|---|---|
| Uniform averaging | [`summary/average_results.csv`](summary/average_results.csv) | [`average/`](average/) |
| Target-Bit Margin | [`summary/targeted_hits.csv`](summary/targeted_hits.csv) | [`targeted/`](targeted/) |
| Majority/bit composition | [`summary/bit_composition.csv`](summary/bit_composition.csv) | [`average/k8/`](average/k8/) |
| One-bit mixture paths | [`one_bit/paths.csv`](one_bit/paths.csv) | [`one_bit/pairs/`](one_bit/pairs/) and [`summary/one_bit_examples.csv`](summary/one_bit_examples.csv) |
| Confidence screening | [`summary/confidence_screening.csv`](summary/confidence_screening.csv) | [`supplementary/confidence_screening/`](supplementary/confidence_screening/) |
| PESQ/STOI/ViSQOL/SI-SDR/SNR | [`supplementary/quality/summary_by_system_k.csv`](supplementary/quality/summary_by_system_k.csv) | [`supplementary/quality/all_trials.csv`](supplementary/quality/all_trials.csv) |
| Temporal offsets | [`supplementary/alignment/summary_direction_averaged.csv`](supplementary/alignment/summary_direction_averaged.csv) | [`supplementary/alignment/all_trials.csv`](supplementary/alignment/all_trials.csv) |
| MP3 and Opus | [`supplementary/codec/summary_by_system_codec.csv`](supplementary/codec/summary_by_system_codec.csv) | [`supplementary/codec/all_trials.csv`](supplementary/codec/all_trials.csv) |
| Registry occupancy | [`supplementary/registry_occupancy/registry_occupancy_k2_average.csv`](supplementary/registry_occupancy/registry_occupancy_k2_average.csv) | [`supplementary/registry_occupancy/registry_occupancy_by_system_k.csv`](supplementary/registry_occupancy/registry_occupancy_by_system_k.csv) |

## Directory structure

| Directory | Purpose | Released scope |
|---|---|---:|
| [`average/`](average/) | Uniform averaging outcomes | 5 systems × 4 K values × 300 trials |
| [`coalitions/`](coalitions/) | Shared valid K=5 and K=8 coalitions | 600 trials |
| [`targets/`](targets/) | Ten selected Target-Bit Margin targets | 600 trial files |
| [`targeted/`](targeted/) | Exact target attempts | 10 files × 3,000 attempts |
| [`one_bit/`](one_bit/) | Valid endpoint pairs and per-trial path summaries | 300 pairs per evaluated system |
| [`confidence/`](confidence/) | Compact K=8 minimum-confidence records | 5 system files |
| [`summary/`](summary/) | Small inputs for manuscript tables and figures | 7 CSV files |
| [`supplementary/`](supplementary/) | Quality, screening, offset, codec, and registry evidence | 17 files |
| [`manifest.csv`](manifest.csv) | Size, row count, and SHA-256 for every released result | 2,757 entries |

The four 16-bit systems use the same coalition in each K=5 and K=8 trial.
TimbreWM uses separately validated 10-bit coalitions. Source paths are relative
to the repository; source speech and personalized-copy caches are not
distributed.

## Record hierarchy

```text
summary CSV
    └── compact number used by a paper table or figure
trial-level CSV or JSON
    ├── source recording and coalition payloads
    ├── source-copy validity
    ├── attack condition and decoded outcome
    └── quality and decoder evidence, when applicable
manifest.csv
    └── checksum and row count for every released data file
```

## Core field definitions

| Field | Meaning |
|---|---|
| `k` | Coalition size |
| `clip_index` | One-based utterance number for a speaker |
| `valid_copy_count` | Coalition copies that decoded to their assigned payload before mixing |
| `payloads_tested` | Candidate payloads checked while constructing a valid coalition |
| `escaped` | Decoded payload matches no coalition member |
| `tracing_failure_pct` | Percentage of trials with `escaped = 1` |
| `target_hit` | Complete decoded payload equals `target_payload` |
| `hits_out_of_10` | Exact hits among the ten selected targets in one trial |
| `selection_score` | Optimized weakest-bit score used to rank target candidates |
| `target_margin` | Decoder margin of the evaluated output, not the target-selection score |

The public method name is **Target-Bit Margin**. Paths and CSV values use
`target_bit_margin`. Display conditions use **Single**, **Average**, and
**Targeted**; CSV values use `single`, `average`, and `targeted`.

Metric columns use lowercase names: `pesq`, `stoi`, `visqol`, `si_sdr`, and
`snr`. Quality metrics compare the mixture with the first valid personalized
copy, not with the unwatermarked source.

## Confidence-data boundary

[`confidence/`](confidence/) contains the compact minimum-confidence records
used by the distribution figure. The five-row paper result is
[`summary/confidence_screening.csv`](summary/confidence_screening.csv).
[`supplementary/confidence_screening/`](supplementary/confidence_screening/)
releases the three sufficient statistics for every evaluated output, the
speaker-fold assignment, and all 25 fold-specific threshold sets, which are
enough to reproduce every screening rate without releasing per-bit vectors.

## Codec controls

The codec records include `valid_post_codec_copy_count` and
`all_post_codec_copies_valid`. These fields separate failures already present
in an individually coded copy from failures that appear after the coded copies
are averaged. The codec summary reports both the full 300-trial result and the
subset in which all five post-codec copies still decode correctly.

## One-bit-data boundary

[`one_bit/paths.csv`](one_bit/paths.csv) records whether each complete path
stays on the two endpoint payloads and how many distinct outputs appear.
[`summary/one_bit_examples.csv`](summary/one_bit_examples.csv) contains every
point used by the manuscript figure. Full per-weight decoder vectors for all
600 paths are not included in this snapshot; the exact commands for
regenerating them under `results/` are in the
[reproduction guide](../REPRODUCIBILITY.md#7-one-bit-paths-and-confidence-screening).

## Verify every record

From the repository root:

```bash
python scripts/verify_release.py
```

The validator checks counts, schemas, aggregates, shared-coalition constraints,
public naming, demos, and all hashes in [`manifest.csv`](manifest.csv).

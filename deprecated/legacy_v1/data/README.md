# Published experiment data

Generated from completed runtime outputs at 2026-08-23 05:11:20 CST.

`data/` is the canonical, version-controlled result tree. Runtime checkpoints and logs
remain under the ignored local `results/` directory. Audio datasets, caches, and model
weights are not included.

Matched `N=1024` experiments retain native payload bits and restrict only the independently
sampled active candidate registry; they do not truncate 16-bit payloads.

## Categories

- `attack/`: 20 CSV files
- `attack_evasion/`: 24 CSV files
- `baselines/`: 20 CSV files
- `bdb/`: 20 CSV files
- `codec_sensitivity/`: 5 CSV files
- `detector_oracle/`: 4 CSV files
- `dm/`: 20 CSV files
- `dm_restart_stability/`: 20 CSV files
- `ecc/`: 20 CSV files
- `eep/`: 20 CSV files
- `evidence_chain/`: 20 CSV files
- `framing_hull/`: 20 CSV files
- `mechanism_diag/`: 1 CSV files
- `other/`: 73 CSV files
- `pgr/`: 20 CSV files
- `pilot/`: 10 CSV files
- `pulse_noise/`: 20 CSV files
- `quality_presence/`: 20 CSV files
- `registry_control/`: 76 CSV files
- `rp/`: 20 CSV files
- `tamper/`: 32 CSV files
- `tamper_arbitrary/`: 20 CSV files
- `tamper_arbitrary_detail/`: 5 CSV files
- `tamper_arbitrary_matched_n1024/`: 20 CSV files
- `tamper_single_copy/`: 9 CSV files
- `tamper_softmin_ablation/`: 3 CSV files
- `tamper_softmin_pilots/`: 10 CSV files
- `tamper_softmin_v2_n1024/`: 20 CSV files
- `tamper_softmin_v2_native/`: 16 CSV files
- `temporal_sensitivity/`: 5 CSV files

Total: 593 completed CSV files.

See `INDEX.csv` for row counts, schemas, byte sizes, runtime source paths, and SHA-256 checksums.

## Merged v5–v8 experimental archive

`paper_v5_v8_merged_20260828/` contains experimental artifacts recovered from
the removed v5–v8 manuscript trees. It has its own `SOURCE_MANIFEST.csv` and
`SHA256SUMS.csv` because it includes CSV, JSON, SVG, PDF, and audit records, not
only the canonical completed CSV format indexed above. Use
`corrected_statistics_20260826/` for final statistical results;
`superseded_statistics_20260825/` is retained only for provenance.

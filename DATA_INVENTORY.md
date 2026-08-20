# Experiment data inventory

Generated: 2026-08-20 19:24:06 CST.

The canonical publication results are stored in `data/`. Local `results/` is runtime-only
and is intentionally not tracked. The audio dataset under `dataset/collusion_300/` is also
excluded from Git.

The main protocol uses 100 bilingual speakers (50 English and 50 Chinese), three 10-second
clips per speaker, K in {2,3,5,8}, and 300 deterministic trials unless a pilot or robustness
experiment explicitly specifies otherwise.

Matched-registry controls use N=1024 independently sampled native identities, include every
coalition member, and preserve all native payload bits.

## Published categories

| Category | Files |
|---|---:|
| `attack` | 20 |
| `baselines` | 20 |
| `bdb` | 20 |
| `codec_sensitivity` | 5 |
| `detector_oracle` | 4 |
| `dm` | 20 |
| `dm_restart_stability` | 20 |
| `ecc` | 20 |
| `eep` | 20 |
| `evidence_chain` | 20 |
| `framing_hull` | 20 |
| `mechanism_diag` | 1 |
| `pgr` | 20 |
| `pilot` | 10 |
| `pulse_noise` | 20 |
| `quality_presence` | 20 |
| `registry_control` | 20 |
| `rp` | 20 |
| `tamper` | 20 |
| `tamper_arbitrary` | 20 |
| `tamper_arbitrary_detail` | 5 |
| `tamper_arbitrary_matched_n1024` | 16 |
| `temporal_sensitivity` | 5 |

Total: 366 completed CSV files.

See `data/INDEX.csv` for the complete machine-readable inventory.

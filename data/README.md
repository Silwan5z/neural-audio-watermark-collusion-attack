# Released data

Only records used by the current study are tracked here.

| Directory | Contents | Expected records |
|---|---|---:|
| `average/` | Uniform averages at K=2, 3, 5, and 8 | 20 system/K cells x 300 trials |
| `coalitions/` | Valid K=5 and K=8 coalitions | 600 trials |
| `targets/` | Ten Payload Match and Bit Margin targets per trial | 1,200 files |
| `targeted/` | Exact-match evaluation for both target methods | 20 files x 3,000 attempts |
| `one_bit/` | Valid one-bit pairs and path results | 300 pairs per system |
| `confidence/` | K=8 bit-confidence records | 5 files |
| `summary/` | Compact table and figure inputs | 6 CSV files |

The four 16-bit systems use the same coalition in each K=5 and K=8 trial.
TimbreWM uses a separately validated 10-bit coalition. Source paths are stored
relative to the repository; speech and marked-copy caches are not distributed.

Run `python scripts/verify_release.py` from the repository root to check all
counts, schemas, key aggregates, shared-coalition constraints, and checksums.
The file list and checksums are stored in `manifest.csv`.

## Naming

Public names follow one rule across paths, code, and data:

- Human-readable method names are **Payload Match** and **Bit Margin**; file
  values are `payload_match` and `bit_margin`.
- `k` is the coalition size. `clip_index` is the 1-based utterance number for
  one speaker. No second public field names the same utterance.
- `escaped` is a trial outcome. `tracing_failure_pct` is the percentage of
  escaped trials.
- `target_hit` means that the complete decoded payload equals
  `target_payload`. `hits_out_of_10` counts those hits among the ten targets.
- `selection_score` ranks targets within one method; it is not compared across
  Payload Match and Bit Margin. `target_margin` describes the decoded output,
  not the Bit Margin selection method.
- `valid_copy_count` counts coalition copies that decode to their assigned
  payload before mixing. `payloads_tested` counts payloads checked while
  constructing a valid coalition.
- `Single`, `Average`, and `Targeted` are display labels. CSV condition values
  use `single`, `average`, and `targeted`.

Metric columns use lowercase names (`pesq`, `stoi`, and `si_sdr`). Paths use
the order `experiment/k/method/system` when all four levels are present; for
example, `targeted/k8/bit_margin/audioseal.csv`.

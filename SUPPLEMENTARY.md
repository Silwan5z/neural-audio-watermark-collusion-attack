# Supplementary evaluations

This release adds three compact analyses requested during manuscript review.
They use the same five watermarking systems and the released 300-recording
evaluation set.  The complete trial records are included so that aggregate
values can be checked without model inference.

## Uniform-mixture quality

`data/supplementary/quality/` adds SI-SDR and SNR to the existing PESQ and
STOI measurements for K=2, 3, 5, and 8.  All waveform metrics compare the
uniform mixture with the first valid personalized coalition copy.  Native-rate
signals are converted to 16 kHz only for metric computation.

The per-system means across 1,200 trials are:

| System | PESQ | STOI | SI-SDR (dB) | SNR (dB) |
|---|---:|---:|---:|---:|
| AudioSeal | 4.5895 | 0.9989 | 35.1857 | 35.2803 |
| WavMark | 4.5462 | 0.9985 | 42.6571 | 42.7508 |
| TimbreWM | 4.5501 | 0.9972 | 32.0119 | 32.1078 |
| VoiceMark | 4.1334 | 0.9765 | 15.2801 | 15.3494 |
| WMCodec | 4.2959 | 0.9856 | 14.7852 | 15.0505 |

For K=2, 3, and 5, the released SI-SDR is retained in the table aggregate;
K=8 SI-SDR and all SNR values are reconstructed.  Four systems reproduce the
released SI-SDR within 0.005 dB.  WMCodec has a mean absolute difference of
0.463 dB because the earlier files did not retain coalition payloads and valid
coalition reconstruction can select replacements.  The WMCodec SNR therefore
describes a fresh valid-coalition reconstruction rather than the exact earlier
waveforms.  Full diagnostics are in `reproduction_audit.json`.

To regenerate runtime records and summaries:

```bash
python scripts/compute_uniform_quality.py --model audioseal
python scripts/summarize_uniform_quality.py
```

The computation can be sharded with `--shard-id` and `--num-shards`.

## Temporal misalignment

The K=5 stress test shifts one coalition member before averaging.  The shifted
member rotates by trial, both time directions are evaluated, and the paired
zero-shift condition uses the same newly reconstructed coalition and recording.

| Condition | TF (%) | PESQ | STOI | SI-SDR (dB) | SNR (dB) |
|---|---:|---:|---:|---:|---:|
| Aligned | 96.67 | 4.408 | 0.990 | 27.37 | 27.51 |
| +/-10 ms | 97.73 | 4.030 | 0.976 | 11.10 | 10.45 |
| +/-20 ms | 97.63 | 3.366 | 0.971 | 11.00 | 10.28 |
| +/-50 ms | 96.67 | 2.372 | 0.963 | 11.03 | 10.36 |

Temporal offsets change mean TF by at most 1.07 percentage points relative to
the paired aligned condition.  However, quality already degrades at 10 ms;
these results do not support a claim that degradation begins only at 50 ms.
This is a one-member shift stress test, not a complete model of independently
processed copies.

The experiment requires the quality reconstruction records because it reuses
their validated K=5 coalitions:

```bash
python scripts/run_alignment_current.py --model audioseal
python scripts/summarize_alignment_current.py
```

## Partial registry occupancy

This is an analytic deployment analysis rather than an additional watermark
inference experiment.  Each registry includes all coalition members and a
uniform random subset of other payloads, and attribution uses exact payload
lookup.  Under those assumptions, the released native escape rate is split
between a registered nonmember and an unassigned output.

For K=2, averaged across the five systems:

| Occupancy | Coalition trace (%) | Registered nonmember (%) | Unassigned (%) | Escape (%) |
|---:|---:|---:|---:|---:|
| 1% | 5.00 | 0.91 | 94.09 | 95.00 |
| 10% | 5.00 | 9.46 | 85.54 | 95.00 |
| 50% | 5.00 | 47.48 | 47.52 | 95.00 |
| 100% | 5.00 | 95.00 | 0.00 | 95.00 |

Sparse registries therefore reduce wrongful attribution to a registered user,
but do not restore coalition traceability: most escaped outputs instead become
unassigned.  This split should not be presented as an independent empirical
attack result.  Structured codes or nearest-neighbor attribution require a
separate experiment.

Regenerate all five-system, all-K values with:

```bash
python scripts/analyze_registry_occupancy.py
```

## Released files

- `all_trials.csv` files contain the complete quality and alignment records.
- `summary_by_system_k.csv` and `summary_by_system_shift.csv` retain the full
  per-system breakdown.
- `summary_direction_averaged.csv` averages positive and negative offsets.
- Registry files contain both observed and bit-space ideal-reference splits.
- Runtime shards and superseded checkpoints remain excluded from Git.

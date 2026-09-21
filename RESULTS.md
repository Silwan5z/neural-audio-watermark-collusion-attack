# Results and complete tables

[Back to the project overview](README.md) ·
[Inspect the released data](data/README.md) ·
[Listen to examples](demos/README.md) ·
[Reproduce the experiments](REPRODUCIBILITY.md)

This is the single numerical-results page for the repository. The first section
reproduces the four principal manuscript tables; the second contains the
additional quality, temporal-offset, codec, and registry analyses. Values are
derived from the released records rather than copied from runtime logs.

**Main paper:** [Uniform averaging](#uniform-averaging) ·
[Coalition-bit behavior](#coalition-bit-behavior-at-k8) ·
[One-bit mixture paths](#one-bit-mixture-paths) ·
[Target-Bit Margin](#target-bit-margin) ·
[Confidence screening](#confidence-screening)

**Additional evidence:** [Temporal offsets](#temporal-offsets) ·
[MP3 and Opus](#independent-lossy-coding) ·
[Registry occupancy](#partial-registry-occupancy) ·
[Result-to-data map](#result-to-data-map)

## How to read the results

All empirical cells use 300 distinct ten-second recordings from 100 speakers,
unless explicitly identified as an analytic calculation. Every personalized
input copy is decoded correctly before it is admitted to a coalition.

- **TF (tracing failure):** the decoded native payload matches no coalition
  member. It measures coalition escape, not necessarily assignment to a real
  registered innocent user.
- **Target success:** the complete decoded payload equals the selected
  nonmember payload.
- **Quality reference:** the first valid personalized copy in the coalition,
  evaluated at 16 kHz after native-rate decoding.

### Evaluation scope

| System | Payload bits | Decoder type | Native sample rate |
|---|---:|---|---:|
| AudioSeal | 16 | Bit-based | 16 kHz |
| WavMark | 16 | Bit-based | 16 kHz |
| TimbreWM | 10 | Bit-based | 22.05 kHz |
| VoiceMark | 16 | Latent-based | 16 kHz |
| WMCodec | 16 | Latent-based | 24 kHz |

The evaluation set contains three clips from each speaker, evenly split between
AISHELL-3 and LibriSpeech. Embedding, mixing, and decoding remain at each
model's native sample rate.

## Main-paper results

### Uniform averaging

The manuscript table contains TF, PESQ, and STOI. ViSQOL, SI-SDR, and SNR are
included here at the same per-system/per-K granularity so that the public table
does not hide the additional quality evidence behind an across-K average.

| System | K | TF (%) | PESQ | STOI | ViSQOL | SI-SDR (dB) | SNR (dB) |
|---|---:|---:|---:|---:|---:|---:|---:|
| AudioSeal | 2 | 99.0 | 4.6040 | 0.9992 | 4.9575 | 36.6724 | 36.7667 |
| AudioSeal | 3 | 97.3 | 4.5910 | 0.9991 | 4.9426 | 35.5503 | 35.6447 |
| AudioSeal | 5 | 98.3 | 4.5856 | 0.9988 | 4.9270 | 34.5779 | 34.6725 |
| AudioSeal | 8 | 100.0 | 4.5773 | 0.9986 | 4.9221 | 33.9423 | 34.0372 |
| WavMark | 2 | 96.7 | 4.5716 | 0.9990 | 4.8807 | 44.1468 | 44.2403 |
| WavMark | 3 | 97.7 | 4.5563 | 0.9987 | 4.8644 | 43.0046 | 43.0986 |
| WavMark | 5 | 98.0 | 4.5391 | 0.9983 | 4.8449 | 42.0950 | 42.1885 |
| WavMark | 8 | 99.3 | 4.5179 | 0.9980 | 4.8350 | 41.3821 | 41.4756 |
| TimbreWM | 2 | 86.3 | 4.5638 | 0.9981 | 4.7818 | 33.5323 | 33.6273 |
| TimbreWM | 3 | 84.7 | 4.5564 | 0.9975 | 4.7298 | 32.3179 | 32.4134 |
| TimbreWM | 5 | 90.7 | 4.5468 | 0.9970 | 4.6853 | 31.4403 | 31.5365 |
| TimbreWM | 8 | 92.0 | 4.5335 | 0.9964 | 4.6389 | 30.7571 | 30.8540 |
| VoiceMark | 2 | 93.7 | 4.2036 | 0.9827 | 3.6921 | 16.5841 | 16.6131 |
| VoiceMark | 3 | 99.0 | 4.1675 | 0.9782 | 3.3889 | 15.4210 | 15.4912 |
| VoiceMark | 5 | 99.3 | 4.1134 | 0.9741 | 3.2211 | 14.5981 | 14.6775 |
| VoiceMark | 8 | 98.7 | 4.0490 | 0.9709 | 3.1170 | 14.5172 | 14.6159 |
| WMCodec | 2 | 99.3 | 4.3673 | 0.9894 | 4.7240 | 16.2861 | 16.5056 |
| WMCodec | 3 | 99.3 | 4.3201 | 0.9865 | 4.6465 | 14.9813 | 15.1739 |
| WMCodec | 5 | 100.0 | 4.2609 | 0.9839 | 4.5787 | 14.1475 | 14.4608 |
| WMCodec | 8 | 99.7 | 4.2351 | 0.9827 | 4.5401 | 13.7260 | 14.0618 |

The ideal bit-choice tracing-failure references are:

| Payload size | K=2 | K=3 | K=5 | K=8 |
|---|---:|---:|---:|---:|
| 16-bit | 97.9 | 99.5 | 99.9 | 99.9 |
| 10-bit | 89.1 | 94.3 | 96.8 | 97.4 |

Sources: [`average_results.csv`](data/summary/average_results.csv),
[`summary_by_system_k.csv`](data/supplementary/quality/summary_by_system_k.csv),
and [`ideal_tracing_failure.csv`](data/summary/ideal_tracing_failure.csv).
Complete trial-level records are under [`data/average/`](data/average/) and
[`data/supplementary/quality/`](data/supplementary/quality/).

**Takeaway.** At K=2, only two valid personalized copies produce 86.3–99.3%
tracing failure. The additional metrics show that this is not obtained by
simply destroying the speech signal.

### Coalition-bit behavior at K=8

| System | Bit agreement (%) | Trial agreement (%) |
|---|---:|---:|
| AudioSeal | 84.9 | 20.7 |
| WavMark | 95.0 | 54.3 |
| TimbreWM | 100.0 | 100.0 |
| VoiceMark | 61.9 | 0.7 |
| WMCodec | 62.5 | 1.0 |

Bit agreement is computed within each trial over non-tied positions and then
averaged over the 300 trials. Trial agreement requires every non-tied position
in that trial to follow the coalition majority. The values are reproducible
from the K=8 JSON records under [`data/average/k8/`](data/average/k8/).

**Takeaway.** TimbreWM follows the coalition majority at every non-tied bit in
every trial, yet its complete payload still escapes in 92.0% of K=8 trials.
Bit-wise regularity therefore does not imply user traceability.

### One-bit mixture paths

Each valid pair differs at exactly one payload position. A path is **direct**
when every evaluated mixture weight decodes to one of the two endpoint
payloads; **Other** means at least one weight decodes to neither endpoint.

| System | Direct paths | Paths containing Other |
|---|---:|---:|
| AudioSeal | 299/300 (99.7%) | 1/300 (0.3%) |
| VoiceMark | 113/300 (37.7%) | 187/300 (62.3%) |

Source: [`paths.csv`](data/one_bit/paths.csv). Endpoint construction records
are under [`data/one_bit/pairs/`](data/one_bit/pairs/), and the plotted path
points are in [`one_bit_examples.csv`](data/summary/one_bit_examples.csv).

**Takeaway.** The dominant AudioSeal response stays on the two endpoints,
whereas VoiceMark frequently changes bits shared by both endpoints and produces
another payload. The current released regeneration contains one AudioSeal
outlier, so its aggregate should be cited as 299/300 rather than 300/300.

### Target-Bit Margin

| System | K=5 target success (%) | K=8 target success (%) | PESQ | STOI |
|---|---:|---:|---:|---:|
| AudioSeal | 41.9 | 43.6 | 4.58 | 0.999 |
| WavMark | 51.4 | 54.4 | 4.52 | 0.998 |
| TimbreWM | 78.4 | 92.3 | 4.52 | 0.996 |
| VoiceMark | 1.1 | 0.6 | 4.04 | 0.972 |
| WMCodec | 1.0 | 1.3 | 4.20 | 0.983 |

Each trial evaluates ten nonmember targets, so every system/K cell contains
3,000 attempts. Source: [`targeted_hits.csv`](data/summary/targeted_hits.csv),
with all attempts under [`data/targeted/`](data/targeted/).

**Takeaway.** The bit-based systems are substantially more controllable by this
bit-level attack. The low rates for VoiceMark and WMCodec apply to Target-Bit
Margin specifically and do not prove inherent resistance to latent-aware
targeted attacks.

### Confidence screening

| System | Single accepted (%) | Uniform average rejected (%) | Target success before -> after (%) |
|---|---:|---:|---:|
| AudioSeal | 94.7 | 100.0 | 43.6 -> 2.4 |
| WavMark | 94.7 | 100.0 | 54.4 -> 4.4 |
| TimbreWM | 95.0 | 100.0 | 92.3 -> 0.0 |
| VoiceMark | 94.3 | 77.7 | 0.6 -> 0.3 |
| WMCodec | 95.0 | 88.0 | 1.3 -> 0.7 |

Source: [`confidence_screening.csv`](data/summary/confidence_screening.csv).
Each system/fold receives its own threshold set, calibrated so the conjunction
of all three conditions retains at least 95% of calibration Single outputs.
This is a non-adaptive, preliminary screening result rather than a complete
collusion-resistant defense.

**Takeaway.** Screening detects a strong distribution shift for the evaluated
attacks, but an adaptive attacker that also optimizes acceptance confidence was
not tested.

## Additional evaluations

### Temporal offsets

One coalition copy is shifted before averaging; positive and negative shift
directions are pooled. Values are cross-system means.

| Condition | TF (%) | PESQ | STOI | SI-SDR (dB) | SNR (dB) |
|---|---:|---:|---:|---:|---:|
| Aligned | 96.67 | 4.408 | 0.990 | 27.37 | 27.51 |
| +/-10 ms | 97.73 | 4.030 | 0.976 | 11.10 | 10.45 |
| +/-20 ms | 97.63 | 3.366 | 0.971 | 11.00 | 10.28 |
| +/-50 ms | 96.67 | 2.372 | 0.963 | 11.03 | 10.36 |

Source: [`summary_cross_system.csv`](data/supplementary/alignment/summary_cross_system.csv).

**Takeaway.** Mean TF changes by at most 1.07 percentage points relative to the
paired aligned condition. Audio quality already falls at 10 ms, so these data
do not support the claim that degradation begins only at 50 ms.

### Independent lossy coding

Each coalition copy is independently coded before averaging. Every
system/condition cell contains the same 300 recordings and validated
coalitions as its paired no-codec control.

| System | Codec | TF (%) | PESQ | STOI | SI-SDR (dB) | SNR (dB) |
|---|---|---:|---:|---:|---:|---:|
| AudioSeal | None | 98.33 | 4.5856 | 0.9988 | 34.5778 | 34.6725 |
| AudioSeal | MP3 128 kbps | 98.00 | 4.4134 | 0.9988 | 27.5270 | 23.1204 |
| AudioSeal | Opus 64 kbps | 97.33 | 4.5573 | 0.9982 | 27.4187 | 26.2248 |
| WavMark | None | 98.00 | 4.5391 | 0.9983 | 42.0948 | 42.1885 |
| WavMark | MP3 128 kbps | 99.00 | 4.3617 | 0.9983 | 29.2096 | 23.5058 |
| WavMark | Opus 64 kbps | 98.00 | 4.5193 | 0.9977 | 28.6748 | 27.5676 |
| TimbreWM | None | 88.00 | 4.5468 | 0.9970 | 31.4402 | 31.5365 |
| TimbreWM | MP3 128 kbps | 88.00 | 4.5477 | 0.9970 | 31.4194 | 24.7660 |
| TimbreWM | Opus 64 kbps | 88.00 | 4.5048 | 0.9962 | 25.8671 | 24.8898 |
| VoiceMark | None | 99.33 | 4.1134 | 0.9741 | 14.5980 | 14.6775 |
| VoiceMark | MP3 128 kbps | 99.33 | 3.9083 | 0.9741 | 14.1834 | 13.9984 |
| VoiceMark | Opus 64 kbps | 99.00 | 4.1090 | 0.9738 | 14.3613 | 13.8872 |
| WMCodec | None | 99.67 | 4.2563 | 0.9838 | 14.1443 | 14.4608 |
| WMCodec | MP3 128 kbps | 100.00 | 4.2547 | 0.9838 | 14.1379 | 14.0946 |
| WMCodec | Opus 64 kbps | 100.00 | 4.2407 | 0.9833 | 13.9410 | 13.7422 |

Source: [`summary_by_system_codec.csv`](data/supplementary/codec/summary_by_system_codec.csv),
with all 4,500 rows in [`all_trials.csv`](data/supplementary/codec/all_trials.csv).

**Takeaway.** Mean TF is 96.67% without coding, 96.87% after MP3, and 96.47%
after Opus. Independent lossy coding therefore does not remove the observed
attack effect under these settings.

### Partial registry occupancy

This is an analytic split of native decoder outcomes under uniform random
registry occupancy and exact payload lookup; it is not a second watermark
inference experiment.

| Occupancy | Coalition trace (%) | Registered nonmember (%) | Unassigned (%) | Escape (%) |
|---:|---:|---:|---:|---:|
| 1% | 5.00 | 0.91 | 94.09 | 95.00 |
| 10% | 5.00 | 9.46 | 85.54 | 95.00 |
| 50% | 5.00 | 47.48 | 47.52 | 95.00 |
| 100% | 5.00 | 95.00 | 0.00 | 95.00 |

Source: [`registry_occupancy_k2_average.csv`](data/supplementary/registry_occupancy/registry_occupancy_k2_average.csv).
Sparse registries reduce attribution to a registered nonmember but mainly
convert those outcomes into unassigned outputs; they do not restore coalition
traceability.

## Important interpretation limits

- Temporal-offset results shift one rotating K=5 member and pool the two shift
  directions; they do not model arbitrary asynchronous copies.
- Registry occupancy is an analytic exact-lookup split of existing native
  outcomes, not a second watermark inference experiment.
- Uniform-quality SI-SDR and SNR use reconstructed valid coalitions. Earlier
  K=2/3/5 SI-SDR values are retained where available; see the reproduction
  notes for the WMCodec reconstruction caveat.
- The confidence screen is non-adaptive, and the empirical corpus contains
  speech only.

## Result-to-data map

| Result | Compact source | Detailed or supporting evidence |
|---|---|---|
| Uniform averaging | [`average_results.csv`](data/summary/average_results.csv) | [`data/average/`](data/average/) |
| Coalition-bit behavior | [`bit_composition.csv`](data/summary/bit_composition.csv) | [`data/average/k8/`](data/average/k8/) |
| One-bit mixture paths | [`paths.csv`](data/one_bit/paths.csv) | [`data/one_bit/pairs/`](data/one_bit/pairs/) and [`one_bit_examples.csv`](data/summary/one_bit_examples.csv) |
| Target-Bit Margin | [`targeted_hits.csv`](data/summary/targeted_hits.csv) | [`data/targeted/`](data/targeted/) |
| Confidence screening | [`confidence_screening.csv`](data/summary/confidence_screening.csv) | [`data/confidence/`](data/confidence/) plus regenerated full vectors |
| Complete quality audit | [`summary_by_system_k.csv`](data/supplementary/quality/summary_by_system_k.csv) | [`all_trials.csv`](data/supplementary/quality/all_trials.csv) |
| Temporal offsets | [`summary_direction_averaged.csv`](data/supplementary/alignment/summary_direction_averaged.csv) | [`all_trials.csv`](data/supplementary/alignment/all_trials.csv) |
| MP3 and Opus | [`summary_by_system_codec.csv`](data/supplementary/codec/summary_by_system_codec.csv) | [`all_trials.csv`](data/supplementary/codec/all_trials.csv) |
| Registry occupancy | [`registry_occupancy_k2_average.csv`](data/supplementary/registry_occupancy/registry_occupancy_k2_average.csv) | [`registry_occupancy_by_system_k.csv`](data/supplementary/registry_occupancy/registry_occupancy_by_system_k.csv) |

# Paper results at a glance

This page collects the principal tables from the five-page manuscript in one
place. Values in the four **paper tables** below were cross-checked against the
released trial records rather than copied from runtime logs. The supplementary
tables then expose additional quality, alignment, codec, and registry analyses.

All empirical cells use 300 distinct ten-second recordings from 100 speakers.
Every personalized input copy is decoded correctly before it is admitted to a
coalition. TF denotes tracing failure: the decoded native payload matches none
of the coalition members.

## Evaluation scope

| System | Payload bits | Decoder type | Native sample rate |
|---|---:|---|---:|
| AudioSeal | 16 | Bit-based | 16 kHz |
| WavMark | 16 | Bit-based | 16 kHz |
| TimbreWM | 10 | Bit-based | 22.05 kHz |
| VoiceMark | 16 | Latent-based | 16 kHz |
| WMCodec | 16 | Latent-based | 24 kHz |

The evaluation set contains 300 speech recordings: three clips from each of
100 speakers, evenly split between AISHELL-3 and LibriSpeech. Embedding,
mixing, and decoding remain at each model's native sample rate.

## Paper Table 1: uniform averaging, with the complete quality audit

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

## Paper Table 2: majority agreement at K=8

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

## Paper Table 3: Target-Bit Margin

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

## Paper Table 4: confidence screening at K=8

| System | Single accepted (%) | Uniform average rejected (%) | Target success before -> after (%) |
|---|---:|---:|---:|
| AudioSeal | 94.0 | 100.0 | 43.6 -> 2.4 |
| WavMark | 94.7 | 100.0 | 54.4 -> 4.5 |
| TimbreWM | 95.0 | 100.0 | 92.3 -> 0.0 |
| VoiceMark | 94.0 | 75.3 | 0.6 -> 0.4 |
| WMCodec | 94.3 | 87.0 | 1.3 -> 0.7 |

Source: [`confidence_screening.csv`](data/summary/confidence_screening.csv).
This is a non-adaptive, preliminary screening result rather than a complete
collusion-resistant defense.

## Temporal misalignment at K=5

One coalition copy is shifted before averaging; positive and negative shift
directions are pooled. Values are cross-system means.

| Condition | TF (%) | PESQ | STOI | SI-SDR (dB) | SNR (dB) |
|---|---:|---:|---:|---:|---:|
| Aligned | 96.67 | 4.408 | 0.990 | 27.37 | 27.51 |
| +/-10 ms | 97.73 | 4.030 | 0.976 | 11.10 | 10.45 |
| +/-20 ms | 97.63 | 3.366 | 0.971 | 11.00 | 10.28 |
| +/-50 ms | 96.67 | 2.372 | 0.963 | 11.03 | 10.36 |

Source: [`summary_cross_system.csv`](data/supplementary/alignment/summary_cross_system.csv).

## Independent lossy coding at K=5

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

## Partial registry occupancy at K=2

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

## Additional resources

- [Audio demos](demos/README.md)
- [Supplementary methods and interpretation limits](SUPPLEMENTARY.md)
- [Reproducibility instructions](REPRODUCIBILITY.md)
- [Released-data schema](data/README.md)

# Audio Watermark Collusion Experiments — Results Index

Snapshot: 2026-08-17 22:12 Asia/Shanghai.  This index describes the live
experiment directory; new CSV files are written to `evaluation/` and detailed
per-task stdout/stderr is written to `logs/`.

## Input data

| Item | Configuration |
|---|---|
| Dataset root | `../dataset/collusion_300/` |
| Clips | 300 total: 150 English LibriSpeech train-clean-100 + 150 Chinese AISHELL-3 |
| Speakers | 100 total, 50 per language, 3 clips per speaker |
| Audio format | mono PCM WAV, 16 kHz, exactly 10.0 s / 160,000 samples |
| Metadata | `../dataset/collusion_300/manifest.csv` contains language, speaker, clip, and original-source path |
| Sampling | Trials are deterministically seeded by `(speaker, K, local_trial)` |

## Shared experimental configuration

| Item | Value |
|---|---|
| Models | `audioseal`, `wavmark`, `voicemark`, `wmcodec`, `timbrewm` |
| Coalition sizes | `K = 2, 3, 5, 8` |
| Standard trial count | 300 per `(script, model, K)` configuration |
| Pulse-noise trial count | 50 per `(model, K)`; the script evaluates multiple pulse settings per trial |
| Active GPUs | GPU 0–5 are scheduled dynamically; GPU 6 is intentionally left unused |
| Persistent schedulers | `watermark_full_suite` and `watermark_full_suite_resume` tmux sessions |

## Experiment families

| Prefix / script | What is compared | Normal output rows |
|---|---|---:|
| `attack_` / `attack.py` | Mean vs FWP (farthest waveform pair) blind collusion attack | 600 |
| `attack_ecc_` / `attack_ecc.py` | Uncoded vs one-bit-correcting ECC, each with Mean and FWP | 1,200 |
| `rp_` / `rp.py` | Random-pair blind control | 300 |
| `eep_` / `eep.py` | Energy-extreme-pair blind control | 300 |
| `baselines_` / `baselines.py` | median, minimum, maximum, rand_minmax, copy_paste | 1,500 |
| `dm_` / `blind_distance.py` | Dispersion maximization | 300 |
| `bdb_` / `blind_minimax.py` | Blind distance balancing | 300 |
| `pgr_` / `pgr.py` | Payload-geometry reference (payload-aware diagnostic) | 300 |
| `tamper_` / `framing.py` | Mean vs TCT targeted tampering, 10 targets/trial | 6,000 |
| `pulse_noise_` / `pulse_noise.py` | Pulse-noise control | 600 |

### ECC definition

`attack_ecc.py` simulates the watermark system embedding an ECC codeword, then
ranks only valid codewords at detection time.  It reports `scheme=uncoded` and
`scheme=ecc1` in the same CSV, so ASR differences are directly comparable.

| Native watermark bits | ECC | Information bits | Correctable errors |
|---:|---|---:|---:|
| 16 (AudioSeal, WavMark, VoiceMark, WMCodec) | shortened Hamming `(16,11)` | 11 | 1 bit |
| 10 (TimbreWM) | shortened Hamming `(10,6)` | 6 | 1 bit |

## Outputs completed at this snapshot

All completed CSV files are under [`evaluation/`](evaluation/).

| Family | Completed `(model: K)` |
|---|---|
| `attack` | `audioseal: 2,3,5,8`; `wavmark: 2,3`; `voicemark: 2,3`; `wmcodec: 2,3`; `timbrewm: 2,3` |
| `attack_ecc` | `audioseal: 2,3,5,8`; `wavmark: 2` |
| `rp` | `audioseal: 2,3,5,8`; `wavmark: 2` |
| `eep` | `audioseal: 2,3,5,8`; `wavmark: 2,3` |
| `baselines` | `audioseal: 2,3,5,8`; `wavmark: 2` |
| `dm` | `audioseal: 2,3,5,8`; `wavmark: 2` |
| `bdb` | `audioseal: 2,3,5,8`; `wavmark: 2` |
| `pgr` | `audioseal: 2,3,5,8`; `wavmark: 2` |
| `tamper` | `audioseal: 2,3,5,8` |
| `pulse_noise` | `audioseal: 2,3,5,8`; `wavmark: 2` |

## Jobs active at this snapshot

| Script | Model | K | GPU allocation |
|---|---|---:|---|
| `framing.py` | WavMark | 2 | dynamically assigned |
| `attack_ecc.py` | WavMark | 3 | dynamically assigned |
| `baselines.py` | WavMark | 3 | dynamically assigned |
| `blind_distance.py` | WavMark | 3 | dynamically assigned |

The dynamic queue continues through every listed family, all five models, and
all four K values.  A follow-up tmux session scans `evaluation/` and runs only
any configuration without a completed CSV, covering worker failures safely.

## How to inspect

```bash
cd /private/users/lym/neural-audio-watermark-collusion-attack
find results/evaluation -maxdepth 1 -name '*.csv' | sort
tail -f results/logs/full_<script>_<model>_K<K>.log
tmux attach -t watermark_full_suite
```

Each result CSV contains reproducibility fields `model`, `K`, `spk`,
`local_t`, and `gi`.  Attack-family CSVs additionally provide ASR and quality
measurements (`PESQ`, `STOI`, `SI_SDR`); ECC CSVs add `scheme`, `code_bits`,
`info_bits`, and `codebook_size`.

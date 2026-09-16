# Scripts

## Experiments

- `run_average.py`: native-rate uniform averaging with valid copies at K=2,
  3, or 5.
- `run_average_k8.py`: K=8 native-rate uniform averaging and decoder evidence.
- `validate_average_k8.py`: validates or replaces invalid K=8 source copies.
- `prepare_coalitions.py`: builds valid K=5 and K=8 coalitions.
- `run_targeted.py`: evaluates Target-Bit Margin on ten selected nonmember
  targets.
- `run_one_bit_pairs.py`: constructs valid payload pairs that differ by one bit.
- `run_one_bit_paths.py`: evaluates mixtures between each valid one-bit pair.
- `collect_confidence.py`: records the minimum bit confidence for Single,
  Average, and at most one successful Targeted output per trial. These compact
  records are the input to the confidence-distribution figure.
- `collect_confidence_full.py`: records full bit-confidence vectors for Single,
  Average, and every exact Target-Bit Margin hit at K=8.
- `screen_confidence.py`: performs speaker-disjoint five-fold calibration of
  the minimum, mean, and log-variance confidence screen from the full records.
- `compute_uniform_quality.py`: reconstructs uniform mixtures and computes
  SI-SDR and SNR using the first valid personalized copy as reference.
- `compute_uniform_visqol.py`: invokes the official ViSQOL binary in speech
  mode on 16 kHz PCM copies of the same reference-mixture pairs.
- `summarize_uniform_quality.py`: merges checkpoints and produces per-system
  and per-coalition-size quality summaries.
- `run_alignment_stress_test.py`: tests K=5 averaging after shifting one rotating
  coalition member by 10, 20, or 50 ms in either direction.
- `summarize_alignment_stress_test.py`: verifies and summarizes all alignment
  cells, including paired zero-shift controls.
- `run_codec_stress_test.py`: independently round-trips every K=5 personalized
  copy through MP3 or Opus before native-rate uniform averaging and decoding.
- `summarize_codec_stress_test.py`: verifies and summarizes the paired no-codec,
  MP3 128 kbps, and Opus 64 kbps conditions.
- `analyze_registry_occupancy.py`: analytically splits observed escape into
  registered-nonmember and unassigned outcomes under exact random registries.

`target_bit_margin.py` and `native_audio.py` contain shared helper functions
used by these entry points.

All experiment entry points embed, mix, and decode at the model's native rate:
16 kHz for AudioSeal, WavMark, and VoiceMark; 22.05 kHz for TimbreWM; and
24 kHz for WMCodec. PESQ, STOI, SI-SDR, and ViSQOL receive 16 kHz copies made
only after the native-rate decoder evaluation.

## Release tools

- `merge_targeted.py`: merges final targeted shards and verifies 300 trials with
  ten targets per trial.
- `summarize_one_bit.py`: reduces full one-bit paths to released results.
- `verify_release.py`: checks schemas, counts, aggregates, and checksums.
- `build_data_manifest.py`: rebuilds `data/manifest.csv`.
- `figures/`: builds the three data-driven figures in `outputs/figures/`.

Runtime output belongs under ignored `results/`. The directory layout follows
`experiment/k/method/system` wherever those levels apply. Only verified
records belong under `data/`.

The full confidence workflow can be sharded independently of the targeted
experiment. For example:

```bash
for shard in 0 1 2 3 4 5 6; do
  python scripts/collect_confidence_full.py \
    --model timbrewm --shard-id "$shard" --num-shards 7 &
done
wait
python scripts/screen_confidence.py --model timbrewm
```

The collector resumes complete trials from its `.partial.csv` file. The screen
writes fold thresholds, aggregate metrics, and the exact speaker allocation to
`results/confidence_screening/`.

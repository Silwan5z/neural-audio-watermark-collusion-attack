# Scripts

## Experiments

- `run_average.py`: uniform averaging with valid copies at K=2, 3, or 5.
- `run_average_k8.py`: K=8 uniform averaging with native decoder evidence.
- `validate_average_k8.py`: validates or replaces invalid K=8 source copies.
- `prepare_coalitions.py`: builds valid K=5 and K=8 coalitions.
- `run_targeted.py`: evaluates Payload Match and Bit Margin on ten selected
  nonmember targets.
- `run_one_bit_pairs.py`: constructs valid payload pairs that differ by one bit.
- `run_one_bit_paths.py`: evaluates mixtures between each valid one-bit pair.
- `collect_confidence.py`: records bit confidence for Single, Average, and
  successful Targeted outputs.

`payload_match.py`, `bit_margin.py`, and `native_audio.py` contain shared helper
functions used by these entry points.

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

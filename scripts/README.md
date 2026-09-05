# Paper scripts

## Main experiments

- `attack.py`: source-correct uniform averaging for K=2, 3, and 5.
- `run_k8_population_native.py` and `complete_k8_source_correct_population.py`: K=8 native decoding and source-correct completion.
- `prepare_shared4_coalitions.py`: coalitions shared by AudioSeal, WavMark, VoiceMark, and WMCodec.
- `run_mrc_pm_native_300clips_shard.py`: PM and MRC evaluation for ten selected nonmember targets.
- `run_onebit_k5_pair_analysis.py` and `run_mixture_path_adaptive.py`: valid one-bit endpoints and continuous mixture paths.
- `collect_identity_bit_confidence_k8.py`: minimum bit confidence for Single, Average, and successful MRC outputs.

`framing.py`, `mrc_solver.py`, `registry_size_control.py`, and the two
`run_k8_constructed_payload_case*` modules provide the optimization and native
decoder helpers imported by the main entry points.

## Release and figures

- `merge_targeted_results.py`: merges seven final shards and validates 300 trials × 10 target attempts.
- `summarize_one_bit_paths.py`: reduces large trajectory files to the per-trial evidence used in the paper.
- `verify_release.py`: read-only integrity and aggregate check.
- `build_data_manifest.py`: regenerates `data/MANIFEST.csv`.
- `figures/`: builders for the three data-driven plots. Generated files are
  written to the ignored `outputs/figures/` directory.

Runtime output belongs under ignored `results/`; only verified final records are copied into `data/`.

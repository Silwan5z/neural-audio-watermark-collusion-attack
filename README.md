# Neural Audio Watermark Collusion

Code and verified experiment records for studying how waveform averaging
affects recipient tracing in neural audio watermarking. The repository covers
AudioSeal, WavMark, TimbreWM, VoiceMark, and WMCodec on 300 ten-second
utterances from 100 speakers.

The manuscript draft is intentionally not distributed in this repository.
Bibliographic records and locally available background papers are kept under
`references/`.

## Repository structure

```text
data/        Verified records and compact summaries used by the study
scripts/     Experiment, aggregation, verification, and plotting entry points
src/         Shared watermark and recipient-registry interfaces
third_party/ Required model adapters under their original licenses
dataset/     Dataset preparation notes; speech files are not distributed
references/  Bibliography and available background papers
deprecated/  Notice for the local, Git-ignored legacy archive
```

Model weights, speech files, runtime caches, logs, checkpoints, generated plots,
and manuscript drafts are excluded from Git.

## Released evidence

| Result | Records | Main entry point |
|---|---|---|
| Uniform averaging at K=2, 3, and 5 | `data/main/` | `scripts/attack.py` |
| K=8 tracing and bit behavior | `data/k8/raw/` | `scripts/run_k8_population_native.py` |
| PM and MRC exact target matches | `data/targeted/` | `scripts/run_mrc_pm_native_300clips_shard.py` |
| One-bit mixture paths | `data/one_bit/` | `scripts/run_onebit_k5_pair_analysis.py` |
| K=8 minimum bit confidence | `data/confidence/` | `scripts/collect_identity_bit_confidence_k8.py` |

`data/coalitions/` stores the validated coalitions and target-selection cache.
The four 16-bit systems use the same K=5 and K=8 coalitions; TimbreWM uses its
own validated 10-bit coalitions. `data/summary/` contains the compact inputs for
tables and plots. Every released file is listed with its size, row count, and
SHA-256 checksum in `data/MANIFEST.csv`.

## Verify the release

Install a CUDA-compatible PyTorch build and the remaining dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The integrity check reads existing records and does not run model inference:

```bash
python scripts/verify_release.py
```

Regenerate the three data-driven plots locally:

```bash
python scripts/figures/build_composition.py
python scripts/figures/build_paths.py
python scripts/figures/build_confidence.py
```

Plots are written to the ignored `outputs/figures/` directory.

## Result definitions

- A mixture **escapes tracing** when its complete decoded payload matches no
  coalition member.
- **Tracing failure (TF)** is the percentage of trials that escape. TF is a
  metric, not an attack.
- A targeted hit requires the complete decoded payload to equal a selected
  nonmember target exactly.
- PM and MRC report the mean number of exact hits among ten selected targets on
  a direct 0--10 scale.

## Maintenance

- Add public result files only under the whitelisted `data/` subdirectories.
- Run `scripts/build_data_manifest.py` after changing released data.
- Run `scripts/verify_release.py` before committing.
- Keep incomplete shards and runtime outputs under ignored `results/`.
- Keep manuscript drafts under ignored `paper/` and superseded work under the
  ignored local `deprecated/legacy_v1/` archive.

## License

Project code is released under the MIT License. Files under `third_party/` and
`references/papers/` retain their upstream licenses or copyright terms.

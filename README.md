# Collusion Breaks Recipient Tracing in Neural Audio Watermarking

This repository contains the manuscript, final experiment records, and the code used for the paper. The release is organized around the five evidence chains that appear in the manuscript; older experiments from the previous project direction are isolated under `deprecated/legacy_v1/`.

## Repository map

```text
paper/       Compilable ICASSP manuscript, figures, tables, and PDF
data/        Final records and summaries used by the manuscript
scripts/     Experiment, aggregation, verification, and figure scripts
src/         Shared registry and watermark wrappers
third_party/ Model adapters retained under their original licenses
dataset/     Dataset instructions only; audio is not distributed
deprecated/  Superseded v1 experiments, scripts, and reference files
```

The current paper evaluates AudioSeal, WavMark, TimbreWM, VoiceMark, and WMCodec on 300 ten-second utterances from 100 speakers. Each speaker contributes three utterances. Every coalition input is required to decode to its assigned payload before mixing.

## Paper evidence

| Manuscript result | Released data | Main script |
|---|---|---|
| Uniform averaging, K=2/3/5 | `data/main/` | `scripts/attack.py` |
| K=8 tracing and bit behavior | `data/k8/raw/` | `scripts/run_k8_population_native.py` |
| PM and MRC targeted matches | `data/targeted/` | `scripts/run_mrc_pm_native_300clips_shard.py` |
| One-bit mixture paths | `data/one_bit/` | `scripts/run_onebit_k5_pair_analysis.py`, `scripts/run_mixture_path_adaptive.py` |
| Minimum bit confidence | `data/confidence/` | `scripts/collect_identity_bit_confidence_k8.py` |

`data/summary/` contains the compact files read by the tables and figure scripts. `data/MANIFEST.csv` records every released data file, its size, row count, and SHA-256 checksum.

## Build and verify

Install Python dependencies after installing a CUDA-compatible PyTorch build:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Verify the released records without model inference:

```bash
python scripts/verify_paper_data.py
```

Regenerate the plotted figures:

```bash
python scripts/figures/build_composition.py
python scripts/figures/build_paths.py
python scripts/figures/build_confidence.py
```

Compile the manuscript from `paper/`:

```bash
tectonic -X compile main.tex
```

Model weights and the underlying speech files are not committed. See [dataset/README.md](dataset/README.md) and [scripts/README.md](scripts/README.md) for the required local layout and experiment entry points.

## Result semantics

- A mixture **escapes tracing** when its complete decoded payload matches no coalition member.
- **Tracing failure (TF)** is the percentage of trials that escape; it is a metric, not an attack.
- A targeted hit counts only when the complete decoded payload exactly equals the selected nonmember target.
- PM and MRC report the mean number of exact hits among ten selected targets on a direct 0--10 scale.

## License

Project code is released under the MIT License. Code under `third_party/` retains the licenses of its upstream projects.

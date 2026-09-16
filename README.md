# Neural Audio Watermark Collusion

Code and verified records for evaluating recipient tracing after multiple
watermarked copies of the same utterance are averaged. The release covers
AudioSeal, WavMark, TimbreWM, VoiceMark, and WMCodec on 300 ten-second
utterances from 100 speakers.

The manuscript draft is not distributed here. Bibliographic records and
background papers used by the project are under `references/`.

**Start here:** [paper tables and headline results](PAPER_RESULTS.md) ·
[audio demos](demos/README.md) ·
[supplementary evaluations](SUPPLEMENTARY.md) ·
[reproducibility](REPRODUCIBILITY.md)

## Headline findings

| Question | Result | Evidence |
|---|---|---|
| Can two valid copies evade tracing? | K=2 TF is 86.3–99.3% across the five systems | [Full uniform-averaging table](PAPER_RESULTS.md#paper-table-1-uniform-averaging-with-the-complete-quality-audit) |
| Can the output be steered to a selected nonmember? | Target-Bit Margin reaches 92.3% on TimbreWM at K=8 | [Targeted table](PAPER_RESULTS.md#paper-table-3-target-bit-margin) |
| Does lossy coding remove the effect? | Mean K=5 TF is 96.67% without coding, 96.87% after MP3, and 96.47% after Opus | [Codec table](PAPER_RESULTS.md#independent-lossy-coding-at-k5) |
| Is confidence screening a complete defense? | No; it is a preliminary, non-adaptive screen | [Screening table and scope](PAPER_RESULTS.md#paper-table-4-confidence-screening-at-k8) |

## Find what you need

| Goal | Start with |
|---|---|
| Read every main-paper number in one place | [`PAPER_RESULTS.md`](PAPER_RESULTS.md) |
| Inspect complete per-trial records | [`data/README.md`](data/README.md) |
| Listen to personalized members and their average | [`demos/README.md`](demos/README.md) |
| Reproduce an experiment | [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) and [`scripts/README.md`](scripts/README.md) |
| Check counts, schemas, aggregates, and hashes | `python scripts/verify_release.py` |

## Repository layout

```text
data/        Verified experiment records and compact table/figure inputs
PAPER_RESULTS.md  All principal manuscript tables in one visible page
demos/       Small listenable examples of members and averaged outputs
dataset/     Dataset manifest and preparation notes; audio is not distributed
scripts/     Experiment, aggregation, plotting, and verification programs
src/         Shared watermark, dataset, and payload-registry interfaces
third_party/ Model adapters and configuration required by the experiments
references/  Bibliography and background papers
```

Model weights, the full speech corpus, caches, logs, checkpoints, generated
plots, and manuscript drafts are excluded from Git. Only the explicitly
documented examples under `demos/` are distributed as audio.

## Released experiments

| Experiment | Released records | Entry point |
|---|---|---|
| Uniform average, K=2, 3, 5, 8 | `data/average/` | `scripts/run_average.py`, `scripts/run_average_k8.py` |
| Targeted mixtures | `data/targeted/` | `scripts/run_targeted.py` |
| One-bit mixture paths | `data/one_bit/` | `scripts/run_one_bit_pairs.py` |
| K=8 confidence distributions | `data/confidence/` | `scripts/collect_confidence.py` |
| K=8 confidence screening | `data/summary/confidence_screening.csv` | `scripts/collect_confidence_full.py`, `scripts/screen_confidence.py` |
| Uniform-mixture quality audit | `data/supplementary/quality/` | `scripts/compute_uniform_quality.py`, `scripts/compute_uniform_visqol.py`, `scripts/summarize_uniform_quality.py` |
| K=5 temporal misalignment | `data/supplementary/alignment/` | `scripts/run_alignment_stress_test.py`, `scripts/summarize_alignment_stress_test.py` |
| K=5 independent codec processing | `data/supplementary/codec/` | `scripts/run_codec_stress_test.py`, `scripts/summarize_codec_stress_test.py` |
| Partial registry occupancy | `data/supplementary/registry_occupancy/` | `scripts/analyze_registry_occupancy.py` |

Target-Bit Margin is the selected-target method reported in the manuscript.
For each trial it optimizes mixture weights for every nonmember payload, ranks
the candidates by the optimized weakest-bit margin, and evaluates the ten
highest-ranked targets. The release also includes Payload Match as an auxiliary
comparison. Its display label in the released CSV is `Bit Margin`, while the
method value and implementation path remain `bit_margin` for compatibility.

`data/coalitions/` stores coalitions whose source copies all decode correctly.
AudioSeal, WavMark, VoiceMark, and WMCodec use the same K=5 and K=8 coalition
in each trial; TimbreWM uses a separately validated coalition because it has a
10-bit payload.
`data/targets/` stores the ten selected targets. Compact inputs for
tables and figures are in `data/summary/`. `data/manifest.csv` records the size,
row count, and SHA-256 checksum of every released result file.

## Definitions

- A mixture **escapes** when its decoded payload matches no coalition member.
- **Tracing failure (TF)** is the percentage of trials that escape. A
  trial-level record therefore uses `escaped`; only an aggregate uses TF.
- A **targeted hit** requires the decoded payload to equal a selected nonmember
  target exactly.
- **Single**, **Average**, and **Targeted** denote a valid single copy, a K=8
  uniform average, and a successful Target-Bit Margin output, respectively.

## Setup and verification

Install a CUDA-compatible PyTorch build, then install the remaining packages:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Verify all released counts, schemas, aggregates, and checksums without model
inference:

```bash
python scripts/verify_release.py
python -m unittest discover -s tests -v
```

Regenerate the three data-driven figures:

```bash
python scripts/figures/build_composition.py
python scripts/figures/build_paths.py
python scripts/figures/build_confidence.py
```

Generated figures are written to the ignored `outputs/figures/` directory.
Exact experiment parameters, native-rate processing, and commands for the
targeted and confidence analyses are documented in
[`REPRODUCIBILITY.md`](REPRODUCIBILITY.md).
The added quality, temporal-misalignment, codec, and registry analyses are documented
in [`SUPPLEMENTARY.md`](SUPPLEMENTARY.md), including interpretation limits.

## Maintenance

- Put incomplete shards and runtime outputs under ignored `results/`.
- Copy only verified final records into the documented `data/` directories.
- Run `python scripts/build_data_manifest.py` after changing released data.
- Run `python scripts/verify_release.py` before committing.
- Keep manuscript drafts under ignored `paper/`.

## License

Project code is released under the MIT License. Third-party code, checkpoints,
and background papers remain subject to their respective upstream terms.

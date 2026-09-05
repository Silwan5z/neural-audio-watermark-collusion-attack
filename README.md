# Neural Audio Watermark Collusion

Code and verified records for evaluating recipient tracing after multiple
watermarked copies of the same utterance are averaged. The release covers
AudioSeal, WavMark, TimbreWM, VoiceMark, and WMCodec on 300 ten-second
utterances from 100 speakers.

The manuscript draft is not distributed here. Bibliographic records and
background papers used by the project are under `references/`.

## Repository layout

```text
data/        Verified experiment records and compact table/figure inputs
dataset/     Dataset manifest and preparation notes; audio is not distributed
scripts/     Experiment, aggregation, plotting, and verification programs
src/         Shared watermark, dataset, and payload-registry interfaces
third_party/ Model adapters and configuration required by the experiments
references/  Bibliography and background papers
```

Model weights, speech files, caches, logs, checkpoints, generated plots, and
manuscript drafts are excluded from Git.

## Released experiments

| Experiment | Released records | Entry point |
|---|---|---|
| Uniform average, K=2, 3, 5, 8 | `data/average/` | `scripts/run_average.py`, `scripts/run_average_k8.py` |
| Targeted mixtures | `data/targeted/` | `scripts/run_targeted.py` |
| One-bit mixture paths | `data/one_bit/` | `scripts/run_one_bit_pairs.py` |
| K=8 bit confidence | `data/confidence/` | `scripts/collect_confidence.py` |

Payload Match and Bit Margin are the two target-selection methods. Payload
Match selects nonmember payloads nearest to the coalition's convex payload
region. Bit Margin selects nonmember payloads for which the weakest target bit
can receive the largest margin. Each method evaluates ten targets and reports
the mean number of exact full-payload hits out of ten.

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
  uniform average, and a successful Bit Margin output, respectively.

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
```

Regenerate the three data-driven figures:

```bash
python scripts/figures/build_composition.py
python scripts/figures/build_paths.py
python scripts/figures/build_confidence.py
```

Generated figures are written to the ignored `outputs/figures/` directory.

## Maintenance

- Put incomplete shards and runtime outputs under ignored `results/`.
- Copy only verified final records into the documented `data/` directories.
- Run `python scripts/build_data_manifest.py` after changing released data.
- Run `python scripts/verify_release.py` before committing.
- Keep manuscript drafts under ignored `paper/`.

## License

Project code is released under the MIT License. Third-party code, checkpoints,
and background papers remain subject to their respective upstream terms.

# Reproduce the evaluation

[Back to the overview](README.md) ·
[Read the results](RESULTS.md) ·
[Inspect the released data](data/README.md) ·
[Browse the scripts](scripts/README.md)

This page separates three tasks that require very different resources:

| Goal | Model weights | Evaluation audio | GPU |
|---|---:|---:|---:|
| Verify the published records and checksums | No | No | No |
| Recompute tables from released records | No | No | No |
| Rerun watermark embedding and decoding | Yes | Yes | Recommended |

Tracked files under `data/` are a fixed release snapshot. Experiment commands
write fresh outputs under the ignored `results/` directory and never silently
replace the published records.

## 1. Verify the published release

From the repository root:

```bash
python scripts/verify_release.py
python -m unittest discover -s tests -v
```

The first command checks all released schemas, trial counts, source-copy
validity, shared coalitions, aggregate numbers, demo files, public naming, and
SHA-256 entries in [`data/manifest.csv`](data/manifest.csv). This is the fastest
way to audit the repository without installing watermark models.

## 2. Prepare a full rerun

Create an environment with a CUDA-compatible PyTorch build and install the
remaining Python dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Then prepare the two external inputs:

1. Follow [`dataset/README.md`](dataset/README.md) to place the 300 fixed
   ten-second WAV files at the paths in the manifest.
2. Follow [`third_party/README.md`](third_party/README.md) to obtain the
   TimbreWM, VoiceMark, and WMCodec weights. AudioSeal and WavMark retrieve
   published weights through their Python packages.

Model weights and source audio are intentionally not redistributed.

## 3. Fixed paper protocol

| Item | Fixed setting |
|---|---|
| Evaluation schedule | 100 speakers × 3 distinct recordings = 300 trials |
| Languages | Mandarin speech from AISHELL-3 and English speech from LibriSpeech |
| Duration | 10 seconds per recording |
| Payload order | Integer payloads converted to LSB-first bit vectors |
| Payload size | 16 bits except TimbreWM, which uses 10 bits |
| Validity gate | Every personalized source copy must decode exactly before mixing |
| Coalition sizes | K = 2, 3, 5, and 8 for uniform averaging |
| Native rates | 16 kHz AudioSeal/WavMark/VoiceMark; 22.05 kHz TimbreWM; 24 kHz WMCodec |
| Quality reference | First valid personalized copy in the coalition |

The registry in the principal experiments is the complete native payload space.
An output **escapes** when its decoded payload matches no coalition member;
tracing failure is the percentage of trials that escape.

## 4. Experiment map

| Paper question | Runner | Released evidence |
|---|---|---|
| Uniform averaging | `run_average.py`, `run_average_k8.py` | [`data/average/`](data/average/) |
| Coalition-bit response | `run_average_k8.py` | [`data/average/k8/`](data/average/k8/) |
| One-bit mixture paths | `run_one_bit_pairs.py`, `run_one_bit_paths.py` | [`data/one_bit/`](data/one_bit/) |
| Target-Bit Margin | `prepare_coalitions.py`, `run_targeted.py` | [`data/targeted/`](data/targeted/) |
| Confidence screening | `collect_confidence_full.py`, `screen_confidence.py` | [`data/summary/confidence_screening.csv`](data/summary/confidence_screening.csv) |
| Complete quality audit | `compute_uniform_quality.py`, `compute_uniform_visqol.py` | [`data/supplementary/quality/`](data/supplementary/quality/) |
| Temporal offsets | `run_alignment_stress_test.py` | [`data/supplementary/alignment/`](data/supplementary/alignment/) |
| MP3 and Opus | `run_codec_stress_test.py` | [`data/supplementary/codec/`](data/supplementary/codec/) |
| Registry occupancy | `analyze_registry_occupancy.py` | [`data/supplementary/registry_occupancy/`](data/supplementary/registry_occupancy/) |

All entry points, outputs, and summarizers are indexed in
[`scripts/README.md`](scripts/README.md).

## 5. Uniform averaging

Mixtures use equal sample-wise weights. K=2, 3, and 5 use
`run_average.py`; K=8 additionally retains native decoder evidence for the
mechanism analysis.

```bash
python scripts/run_average.py --model audioseal --k 2
python scripts/run_average.py --model audioseal --k 5
python scripts/run_average_k8.py --model audioseal
```

The runners support `--shard-id` and `--num-shards`. Embedding, mixing, and
decoding stay at the backend's native sample rate. PESQ, STOI, and SI-SDR
receive 16 kHz evaluation copies only after decoding.

## 6. Target-Bit Margin

Target-Bit Margin is a stronger payload-aware experiment. It knows the
coalition and candidate target payloads but does not use a clean reference,
model parameters, gradients, or iterative decoder queries. For each nonmember,
it optimizes mixture weights to increase the weakest target-bit support, then
decodes the ten highest-ranked candidates once each.

Fixed settings:

- K is 5 or 8, with 300 trials and ten evaluated targets per trial.
- Weights are nonnegative, sum to one, and have effective coalition size of at
  least 0.6K.
- The soft-minimum inverse temperature is 8 and entropy weight is 0.05.
- SLSQP starts from uniform weights with `maxiter=300` and `ftol=1e-14`; a
  failed solve is retried with `maxiter=1000` and `ftol=1e-10`.
- Success requires an exact complete-payload match to the selected nonmember.

The four 16-bit systems share validated coalitions. TimbreWM draws separate
10-bit coalitions. Construct the shared coalitions before launching targeted
shards:

```bash
for k in 5 8; do
  for shard in 0 1 2 3 4 5 6; do
    for model in voicemark wmcodec; do
      python scripts/prepare_coalitions.py --stage validate --pool full \
        --model "$model" --k "$k" --shard-id "$shard" --num-shards 7
    done
    python scripts/prepare_coalitions.py --stage screen --k "$k" \
      --shard-id "$shard" --num-shards 7
    for model in audioseal wavmark voicemark wmcodec; do
      python scripts/prepare_coalitions.py --stage validate --pool candidates \
        --model "$model" --k "$k" --shard-id "$shard" --num-shards 7
    done
    python scripts/prepare_coalitions.py --stage finalize --k "$k" \
      --shard-id "$shard" --num-shards 7
  done
done
```

Run and merge one model/K condition as follows:

```bash
for shard in 0 1 2 3 4 5 6; do
  python scripts/run_targeted.py --method target_bit_margin \
    --model timbrewm --k 8 --shard-id "$shard" --num-shards 7
done
python scripts/merge_targeted.py
```

Small differences of one or two exact hits can occur across SciPy, BLAS, CUDA,
or checkpoint environments when optimized candidates lie close to a decision
boundary. The released CSVs, rather than rounded table entries, are the
reference records.

## 7. One-bit paths and confidence screening

The one-bit experiment constructs 300 valid AudioSeal and VoiceMark payload
pairs that differ at exactly one position, then sweeps the mixture weight. It
tests whether mixing changes only the disputed bit or also changes bits on
which the two users agree.

```bash
python scripts/run_one_bit_pairs.py --model audioseal
python scripts/run_one_bit_pairs.py --model voicemark
python scripts/run_one_bit_paths.py --model audioseal \
  --shard-id 0 --num-shards 1
python scripts/run_one_bit_paths.py --model voicemark \
  --shard-id 0 --num-shards 1
python scripts/summarize_one_bit.py
```

Confidence screening uses complete per-bit evidence for one Single output, one
K=8 uniform Average, and every exact Target-Bit Margin hit. Five
speaker-disjoint folds use 80 speakers for calibration and 20 for testing.
Thresholds are fitted only on valid Single outputs, and an output is accepted
only when its minimum, mean, and log-variance confidence statistics all pass.

```bash
for shard in 0 1 2 3 4 5 6; do
  python scripts/collect_confidence_full.py --model timbrewm \
    --shard-id "$shard" --num-shards 7
done
python scripts/screen_confidence.py --model timbrewm \
  --seed 20260905 --folds 5 --retention 0.95 --z-step 0.001
```

The published screening result is non-adaptive. An attacker that jointly
optimizes target identity and acceptance confidence was not evaluated.

## 8. Additional evaluations

### Quality metrics

`compute_uniform_quality.py` reconstructs uniform mixtures and adds SI-SDR and
SNR. `compute_uniform_visqol.py` uses the official ViSQOL v3.1.0 binary in
speech mode on temporary 16 kHz PCM-16 files.

```bash
python scripts/compute_uniform_quality.py --model audioseal
python scripts/compute_uniform_visqol.py --model audioseal \
  --visqol-bin /path/to/visqol/bazel-bin/visqol \
  --visqol-root /path/to/visqol
python scripts/summarize_uniform_quality.py
```

For K=2, 3, and 5, released SI-SDR values are retained; K=8 SI-SDR and all SNR
values are reconstructed. Four systems reproduce SI-SDR within 0.005 dB.
WMCodec differs by 0.463 dB on average because earlier files did not retain the
coalition payloads and valid reconstruction can select replacements. Its SNR
therefore describes a fresh valid-coalition reconstruction rather than the
exact earlier waveforms. Full diagnostics are released in
[`reproduction_audit.json`](data/supplementary/quality/reproduction_audit.json).

### Temporal offsets

At K=5, one rotating coalition member is shifted by ±10, ±20, or ±50 ms
before averaging. The paired zero-shift condition uses the same reconstructed
coalition and recording.

```bash
python scripts/run_alignment_stress_test.py --model audioseal
python scripts/summarize_alignment_stress_test.py
```

This is a one-member stress test, not a complete model of arbitrary independent
misalignment. Quality already degrades at 10 ms.

### Independent lossy coding

At K=5, every personalized copy is independently round-tripped through MP3 at
128 kbps or Opus at 64 kbps before averaging. The paired `none` condition uses
the same validated coalition and recording.

```bash
python scripts/run_codec_stress_test.py --model audioseal
python scripts/summarize_codec_stress_test.py
```

### Partial registry occupancy

This analysis performs no new model inference. It analytically divides each
observed native escape into registered-nonmember and unassigned outcomes under
uniform random registry occupancy and exact payload lookup.

```bash
python scripts/analyze_registry_occupancy.py
```

The result should not be generalized to structured fingerprinting codebooks or
nearest-neighbor attribution rules, which require separate experiments.

## 9. Release maintenance

After changing any tracked result file:

```bash
python scripts/build_data_manifest.py
python scripts/verify_release.py
```

Only verified final records belong under `data/`. Runtime shards, checkpoints,
logs, and regenerated confidence vectors belong under ignored `results/`.

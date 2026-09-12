# Reproducing the reported experiments

This document records the paper protocol implemented by the current experiment
entry points. Generated audio, model weights, and runtime outputs are excluded
from Git. Tracked records are a fixed release snapshot and are never overwritten
by these commands; fresh outputs are written under `results/`.

## Fixed evaluation set

All reported experiments use 300 ten-second utterances: three clips from each
of 100 speakers. The fixed order is stored in
`dataset/collusion_300/manifest.csv`. Payload integers are converted to
LSB-first bit vectors. AudioSeal, WavMark, VoiceMark, and WMCodec use 16-bit
payloads; TimbreWM uses 10 bits. Each source copy must decode exactly to its
assigned payload before it can enter a coalition.

The registry is the complete native payload space. A decoded payload therefore
always maps to an identity. An escape occurs when it matches no coalition
member, and TF is the percentage of escaped trials.

## Uniform averaging

Coalition sizes are 2, 3, 5, and 8. Mixtures use equal sample-wise weights.
`run_average.py` produces the K=2, 3, and 5 records; `run_average_k8.py`
produces the K=8 records and saves native decoder evidence.

Embedding, mixing, and decoding stay at each backend's native rate: 16 kHz for
AudioSeal, WavMark, and VoiceMark; 22.05 kHz for TimbreWM; and 24 kHz for
WMCodec. The source corpus is loaded at 16 kHz and resampled once before native
embedding when required. PESQ, STOI, and SI-SDR receive 16 kHz evaluation
copies made only after native-rate decoding. `scripts/native_audio.py` provides
the shared embedding cache and decoder path used by all averaging, targeted,
coalition-validation, and confidence entry points.

## Target-Bit Margin

For coalition bits `b_ij`, target bits `t_j`, and mixture weights `a_i`, the
per-bit signed support is

```text
gamma_j = (2 t_j - 1) (sum_i a_i b_ij - 0.5).
```

The implementation uses these fixed settings:

- K is 5 or 8, with 300 trials and ten evaluated targets per trial.
- Every payload outside the coalition is a candidate: `2^d - K` candidates for
  a `d`-bit system.
- Weights are nonnegative and sum to one.
- Effective coalition size is `1 / sum_i a_i^2 >= 0.6 K`.
- The weakest-bit margin is approximated by a soft minimum with `beta = 8.0`.
- An entropy term with weight `0.05` stabilizes weight optimization. It is not
  included in the score used to rank candidate targets.
- SLSQP starts from uniform weights, uses bounds `[0, 1]`, and runs with
  `maxiter=300`, `ftol=1e-14`. A failed solve is retried with
  `maxiter=1000`, `ftol=1e-10`.
- Candidates are ordered by decreasing optimized soft-minimum margin; payload
  integer breaks an exact score tie. The first ten are decoded once each.
- Success requires an exact complete-payload match to the selected nonmember.
  No clean reference, model parameters, or iterative decoder queries are used
  to optimize the weights.

The constants are defined in `scripts/run_targeted.py`, and the optimizer is in
`scripts/bit_margin.py`. Runtime shards are merged with
`scripts/merge_targeted.py`.

For the four 16-bit systems, first construct coalitions that decode exactly in
all four systems under the native-rate protocol. TimbreWM draws and validates
its 10-bit coalitions inside the targeted runner.

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

```bash
for shard in 0 1 2 3 4 5 6; do
  python scripts/run_targeted.py --method bit_margin --model timbrewm \
    --k 8 --shard-id "$shard" --num-shards 7
done
python scripts/merge_targeted.py
```

Small differences of one or two exact hits can occur across SciPy, BLAS, CUDA,
or checkpoint environments when optimized candidates lie close to a decision
boundary. Released CSVs, rather than rounded table entries, are the reference
records.

## Confidence distributions and screening

Bit confidence is the support assigned to the selected bit. AudioSeal and
TimbreWM use native bit probabilities; WavMark uses the fraction of valid
windows voting for one; VoiceMark and WMCodec sum class probabilities
consistent with each bit.

Two collectors have intentionally different roles:

- `collect_confidence.py` saves the minimum confidence and at most one targeted
  hit per trial. It reproduces the compact distribution input in
  `data/confidence/`.
- `collect_confidence_full.py` saves the complete probability and confidence
  vectors for one Single output, one Average output, and every exact
  Target-Bit Margin hit in each K=8 trial. These records are required for the
  three-statistic screen.

`screen_confidence.py` groups all three clips from a speaker in the same fold.
Five speaker-disjoint folds use 80 speakers for calibration and 20 for testing.
Thresholds use training Single outputs only. A common one-sided Gaussian
boundary `z` defines a lower mean threshold, an upper log-variance threshold,
and the matching empirical lower-tail quantile for minimum confidence. The
smallest `z` on a 0.001 grid that jointly accepts at least 95% of training
Single outputs is selected. Variance is the population variance across bits and
is transformed as `log(variance + 1e-12)`. A test output is accepted only when
all three thresholds pass.

```bash
for shard in 0 1 2 3 4 5 6; do
  python scripts/collect_confidence_full.py --model timbrewm \
    --shard-id "$shard" --num-shards 7
done
python scripts/screen_confidence.py --model timbrewm \
  --seed 20260905 --folds 5 --retention 0.95 --z-step 0.001
```

The screen writes the exact fold membership and thresholds alongside its
summary. The paper-level aggregate is released as
`data/summary/confidence_screening.csv`. The original full confidence vectors
and fold allocation were not part of the first public data snapshot, so a fresh
model rerun should be reported as a reproduction rather than silently replacing
the released aggregate.

## Release checks

After changing tracked data, rebuild the checksum manifest and verify all
released schemas and aggregates:

```bash
python scripts/build_data_manifest.py
python scripts/verify_release.py
```

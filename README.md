<div align="center">

# Averaging Collusion Breaks Traceability

### in Personalized Neural Audio Watermarking

Two recipients can average their valid copies and make a watermark decoder
return a different ID—while the recording still sounds almost unchanged.

<br>

<a href="https://silwan5z.github.io/neural-audio-watermark-collusion-attack/index.html?v=2"><strong>▶ Open the interactive audio lab</strong></a>

<sub>Clean and watermarked copies · five systems · K = 2/3/5/8 · timing offsets · MP3 and Opus · quality metrics</sub>

<br><br>

[Results](RESULTS.md) · [Reproduce](REPRODUCIBILITY.md) · [Released data](data/README.md)

</div>

---

## The idea in 20 seconds

Suppose Alice and Bob receive different watermarked copies of the same audio.
Each copy is valid and traces back to its owner. They take the sample-by-sample
average:

```text
Alice's copy ─┐
              ├─ 50/50 waveform average ─→ decoder returns a different ID
Bob's copy ───┘
```

No clean audio is needed. The practical attack uses no model parameters,
gradients, or decoder queries. At coalition size K=2, this simple operation
causes **86.3–99.3% tracing failure** across the five evaluated systems.

The audio lab is the best place to begin. First compare the clean recording,
two valid personalized copies, and their average. Then switch among all five
systems to hear every coalition size, synchronization condition, and codec
setting shown in the demo. Each row pairs one audio clip with its coalition,
decoded ID, PESQ, STOI, and SI-SDR values.

## What should I look at?

| If you want to… | Go here |
|---|---|
| Hear the effect and see exact decoded IDs | **Use the audio lab above** |
| Read the main and supplementary numbers | [RESULTS.md](RESULTS.md) |
| Inspect trial-level evidence | [data/](data/README.md) |
| Re-run an experiment | [REPRODUCIBILITY.md](REPRODUCIBILITY.md) |
| Understand which script does what | [scripts/README.md](scripts/README.md) |

## What the study finds

- **Two copies are already enough.** K=2 tracing failure ranges from 86.3% to
  99.3% across AudioSeal, WavMark, TimbreWM, VoiceMark, and WMCodec.
- **Small timing errors do not remove the effect.** Across the tested aligned
  and ±10/20/50 ms K=5 conditions, mean tracing failure remains 96.7–97.7%.
- **Lossy coding does not remove it either.** The cross-system K=5 means are
  96.67%, 96.87%, and 96.47% without coding, after MP3, and after Opus.
- **Some bit-based decoders can be steered.** The payload-aware Target-Bit
  Margin experiment reaches 92.3% target success on TimbreWM at K=8.
- **Confidence screening helps, but is not a complete defense.** It is reported
  as a preliminary, non-adaptive mitigation rather than a final solution.

These bullets are the story. The paper and [full tables](RESULTS.md) provide the
definitions, protocol, uncertainty, and system-by-system breakdown.

## Evaluation map

| | Scope |
|---|---|
| **Systems** | AudioSeal, WavMark, TimbreWM, VoiceMark, WMCodec |
| **Speech** | 300 ten-second recordings from 100 speakers |
| **Coalitions** | K = 2, 3, 5, and 8 |
| **Validity rule** | Every personalized copy must decode correctly before mixing |
| **Attack input** | Only coalition members’ watermarked waveforms |
| **Released evidence** | Trial records, summaries, checksums, and audio examples |

## Repository layout

```text
demos/                Browser demo and its auditable audio examples
data/                 Released trial records, summaries, and checksums
scripts/              Experiment runners, summaries, figures, and validators
src/                  Shared data, registry, and watermark interfaces
dataset/              Fixed 300-recording evaluation manifest
tests/                Consistency and regression tests
RESULTS.md             Complete public result tables
REPRODUCIBILITY.md     Environment, commands, and protocol details
```

Large speech files, model weights, embedding caches, logs, and manuscript drafts
are intentionally excluded. Fresh runs go to the ignored `results/` directory;
released records under `data/` are not silently overwritten.

## Quick verification

The published records can be checked without a GPU or model weights:

```bash
python scripts/verify_release.py
python -m unittest discover -s tests -v
```

For model inference and complete reruns, follow
[REPRODUCIBILITY.md](REPRODUCIBILITY.md). A locked dependency set is provided in
[`requirements-lock.txt`](requirements-lock.txt).

## Reading the claim carefully

“Tracing failure” means the decoded payload differs from every coalition
member. In a sparse real-world registry, that output might name a registered
nonmember or might remain unassigned; the
[registry analysis](RESULTS.md#partial-registry-occupancy) reports those cases
separately. Uniform averaging is the practical black-box attack. Target-Bit
Margin is a stronger payload-aware experiment and should not be conflated with
it. The empirical scope is speech, not music or environmental audio.

## License

Original project code is released under the [MIT License](LICENSE). Third-party
components, model weights, datasets, and reference papers retain their original
terms; see [third_party/README.md](third_party/README.md),
[dataset/README.md](dataset/README.md), and
[references/README.md](references/README.md).

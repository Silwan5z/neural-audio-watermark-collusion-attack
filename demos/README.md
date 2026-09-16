# Audio demos

[Back to the overview](../README.md) ·
[Read the aggregate results](../RESULTS.md) ·
[Inspect the trial records](../data/README.md)

These examples use the same ten-second LibriSpeech-derived recording and the
same K=2 payload pair across AudioSeal and VoiceMark. They let a listener
compare two valid personalized copies with their 50/50 waveform average.

## Listen in this order

1. Open [`source_reference.wav`](source_reference.wav).
2. Compare the two valid member copies for one system.
3. Open that system's `uniform_average.wav` and compare its sound with the
   members while noting the decoded identity in the table below.

GitHub opens each WAV through its file page; use **View raw** if the browser
does not show an inline audio player.

## Files and decoded outcomes

| File | System | Meaning |
|---|---|---|
| [`source_reference.wav`](source_reference.wav) | none | Evaluation source before watermark embedding |
| [`audioseal/member_24199.wav`](audioseal/member_24199.wav) | AudioSeal | Valid personalized copy for payload 24199 |
| [`audioseal/member_31849.wav`](audioseal/member_31849.wav) | AudioSeal | Valid personalized copy for payload 31849 |
| [`audioseal/uniform_average.wav`](audioseal/uniform_average.wav) | AudioSeal | 50/50 average; decodes to nonmember 23648 |
| [`voicemark/member_24199.wav`](voicemark/member_24199.wav) | VoiceMark | Valid personalized copy for payload 24199 |
| [`voicemark/member_31849.wav`](voicemark/member_31849.wav) | VoiceMark | Valid personalized copy for payload 31849 |
| [`voicemark/uniform_average.wav`](voicemark/uniform_average.wav) | VoiceMark | 50/50 average; decodes to nonmember 60600 |

## Provenance and scope

The example is trial 150, speaker 103, clip 1. The source manifest traces it to
`LibriSpeech/train-clean-100/103/1240/103-1240-0000.flac`. All released demo
files are mono, 16 kHz, PCM-16 WAV. The average is computed sample by sample
from the two corresponding full-length member files. The released PCM-16 files
were decoded again after conversion: both member identities remain valid and
both averages remain coalition escapes.

[`metadata.csv`](metadata.csv) records the payloads, escape outcome, and PESQ,
STOI, and SI-SDR values for both model examples.

These demos illustrate listening quality and one concrete tracing failure; one
example cannot establish an aggregate attack rate. Aggregate TF and quality
results are reported in
[`RESULTS.md`](../RESULTS.md), and the exact trial records are in
[`data/average/k2/`](../data/average/k2/).

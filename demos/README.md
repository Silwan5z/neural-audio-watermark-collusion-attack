# Audio demos

[Open the interactive browser demo](https://silwan5z.github.io/neural-audio-watermark-collusion-attack/) ·
[Back to the overview](../README.md) ·
[Read the aggregate results](../RESULTS.md) ·
[Inspect the trial records](../data/README.md)

These examples use the same ten-second LibriSpeech-derived recording and the
same K=2 trial across all five evaluated systems. The four 16-bit systems use
payloads 24199 and 31849; TimbreWM uses native 10-bit payloads 378 and 497.
For every system, listeners can compare two valid personalized copies with
their 50/50 waveform average.

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
| [`wavmark/member_24199.wav`](wavmark/member_24199.wav) | WavMark | Valid personalized copy for payload 24199 |
| [`wavmark/member_31849.wav`](wavmark/member_31849.wav) | WavMark | Valid personalized copy for payload 31849 |
| [`wavmark/uniform_average.wav`](wavmark/uniform_average.wav) | WavMark | 50/50 average; decodes outside the coalition |
| [`timbrewm/member_378.wav`](timbrewm/member_378.wav) | TimbreWM | Valid personalized copy for payload 378 |
| [`timbrewm/member_497.wav`](timbrewm/member_497.wav) | TimbreWM | Valid personalized copy for payload 497 |
| [`timbrewm/uniform_average.wav`](timbrewm/uniform_average.wav) | TimbreWM | 50/50 average; decodes outside the coalition |
| [`voicemark/member_24199.wav`](voicemark/member_24199.wav) | VoiceMark | Valid personalized copy for payload 24199 |
| [`voicemark/member_31849.wav`](voicemark/member_31849.wav) | VoiceMark | Valid personalized copy for payload 31849 |
| [`voicemark/uniform_average.wav`](voicemark/uniform_average.wav) | VoiceMark | 50/50 average; decodes to nonmember 60600 |
| [`wmcodec/member_24199.wav`](wmcodec/member_24199.wav) | WMCodec | Valid personalized copy for payload 24199 |
| [`wmcodec/member_31849.wav`](wmcodec/member_31849.wav) | WMCodec | Valid personalized copy for payload 31849 |
| [`wmcodec/uniform_average.wav`](wmcodec/uniform_average.wav) | WMCodec | 50/50 average; decodes outside the coalition |

## Provenance and scope

The example is trial 150, speaker 103, clip 1. The source manifest traces it to
`LibriSpeech/train-clean-100/103/1240/103-1240-0000.flac`. All released demo
files are mono, 16 kHz, PCM-16 WAV. Embedding, averaging, and decoding use each
system's native sample rate; the released files are converted to 16 kHz for
consistent browser playback. The K=2 experiment records verify that every
member copy is valid and every displayed average is a coalition escape.

[`metadata.csv`](metadata.csv) records the payloads, escape outcome, and PESQ,
STOI, and SI-SDR values for all five model examples. The original K=2 records
retain the exact average payload for AudioSeal and VoiceMark; for the other
three systems they retain the verified escape indicator but not that raw ID.

These demos illustrate listening quality and one concrete tracing failure; one
example cannot establish an aggregate attack rate. Aggregate TF and quality
results are reported in
[`RESULTS.md`](../RESULTS.md), and the exact trial records are in
[`data/average/k2/`](../data/average/k2/).

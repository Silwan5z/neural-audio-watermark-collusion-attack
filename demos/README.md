# Audio demos

[Open the interactive browser demo](https://silwan5z.github.io/neural-audio-watermark-collusion-attack/index.html?v=2) ·
[Back to the overview](../README.md) ·
[Read the aggregate results](../RESULTS.md) ·
[Inspect the trial records](../data/README.md)

The browser demo uses one ten-second LibriSpeech-derived recording throughout.
Its five switchable views contain the clean source, valid personalized copies,
uniform averages at K = 2/3/5/8, seven timing conditions, and three codec
conditions for every evaluated system. Each condition is presented as one
complete row with its audio, coalition IDs, decoded ID, and quality metrics.

Players load one MP3 only after it is selected. If that request fails, the page
automatically switches the same player to its WAV counterpart. The repository
keeps both formats for direct inspection.

## Files and decoded outcomes

| File | System | Meaning |
|---|---|---|
| [`source_reference.wav`](source_reference.wav) | none | Evaluation source before watermark embedding |
| [`audioseal/member_24199.wav`](audioseal/member_24199.wav) | AudioSeal | Valid personalized copy for payload 24199 |
| [`audioseal/member_31849.wav`](audioseal/member_31849.wav) | AudioSeal | Valid personalized copy for payload 31849 |
| [`audioseal/uniform_average.wav`](audioseal/uniform_average.wav) | AudioSeal | 50/50 average; decodes to nonmember 23648 |
| [`wavmark/member_24199.wav`](wavmark/member_24199.wav) | WavMark | Valid personalized copy for payload 24199 |
| [`wavmark/member_31849.wav`](wavmark/member_31849.wav) | WavMark | Valid personalized copy for payload 31849 |
| [`wavmark/uniform_average.wav`](wavmark/uniform_average.wav) | WavMark | 50/50 average; decodes to 32495 |
| [`timbrewm/member_378.wav`](timbrewm/member_378.wav) | TimbreWM | Valid personalized copy for payload 378 |
| [`timbrewm/member_497.wav`](timbrewm/member_497.wav) | TimbreWM | Valid personalized copy for payload 497 |
| [`timbrewm/uniform_average.wav`](timbrewm/uniform_average.wav) | TimbreWM | 50/50 average; decodes to 504 |
| [`voicemark/member_24199.wav`](voicemark/member_24199.wav) | VoiceMark | Valid personalized copy for payload 24199 |
| [`voicemark/member_31849.wav`](voicemark/member_31849.wav) | VoiceMark | Valid personalized copy for payload 31849 |
| [`voicemark/uniform_average.wav`](voicemark/uniform_average.wav) | VoiceMark | 50/50 average; decodes to nonmember 60600 |
| [`wmcodec/member_24199.wav`](wmcodec/member_24199.wav) | WMCodec | Valid personalized copy for payload 24199 |
| [`wmcodec/member_31849.wav`](wmcodec/member_31849.wav) | WMCodec | Valid personalized copy for payload 31849 |
| [`wmcodec/uniform_average.wav`](wmcodec/uniform_average.wav) | WMCodec | 50/50 average; decodes to 3209 |

## Provenance and scope

The examples are trial 150, speaker 103, clip 1. The source manifest traces it to
`LibriSpeech/train-clean-100/103/1240/103-1240-0000.flac`. All released demo
WAV files are mono, 16 kHz, and PCM-16. Matching MP3 files are provided for
browser compatibility. Embedding, averaging, and decoding use each system's
native sample rate; the browser copies are converted to 16 kHz for consistent
playback. The K=2 experiment records verify that every member copy is valid and
every displayed average is a coalition escape.

[`metadata.csv`](metadata.csv) records the personalized-copy example for each
system. [`conditions/metadata.csv`](conditions/metadata.csv) maps every one of
the 70 condition clips to its evidence record and PESQ, STOI, and SI-SDR
values. Native decoder outputs are stored under [`decoded/`](decoded/).

These demos illustrate listening quality and one concrete tracing failure; one
example cannot establish an aggregate attack rate. Aggregate TF and quality
results are reported in
[`RESULTS.md`](../RESULTS.md), and the exact trial records are in
[`data/average/k2/`](../data/average/k2/).

# Evaluation audio

The study uses `collusion_300/manifest.csv`: 100 speakers with three distinct
10-second, 16 kHz mono utterances per speaker. The English half is derived from
LibriSpeech train-clean-100; the Mandarin half is derived from AISHELL-3. The
manifest fixes the speaker, clip, and source-audio schedule used by every
experiment.

Audio is not redistributed. Place each prepared WAV at the relative `path`
listed in the manifest:

```text
dataset/collusion_300/<language>/<speaker>/<language>_<speaker>_<clip>.wav
```

Each WAV must be mono, 16 kHz, and exactly 10 seconds. `src/registry.py`
validates that all 100 speakers have clip indices 1, 2, and 3 before running an
experiment. The `sources` column records the upstream utterance or utterances
used to construct each fixed-length clip.

Users must obtain LibriSpeech and AISHELL-3 under their original dataset terms.

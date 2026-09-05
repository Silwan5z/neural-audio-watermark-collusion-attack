# Third-party code

This directory contains only the model components imported by the experiment
adapters. They are not covered by the project's MIT License.

| Directory | Upstream project | Upstream terms |
|---|---|---|
| `timbrewm/` | [TimbreWatermarking](https://github.com/TimbreWatermarking/TimbreWatermarking) | GPL-3.0; the bundled HiFi-GAN component includes its own MIT license |
| `voicemark/` | [VoiceMark](https://huggingface.co/spaces/haiyunli/VoiceMark) | GPL, as declared by the upstream Space |
| `wmcodec/` | [WMCodec](https://github.com/zjzser/WMCodec) | Consult the upstream repository; no repository-wide license file is included here |

Model weights are not tracked. Download them from the corresponding upstream
project and place them at these paths:

- `timbrewm/results/ckpt/pth/compressed_none-conv2_ep_20_2023-01-17_23_01_01.pth.tar`
- `timbrewm/hifigan/model/VCTK_V1/generator_v1`
- `voicemark/voicemark.pth`
- `wmcodec/save_model/g_00150000`

AudioSeal and WavMark load their published weights through their Python
packages.

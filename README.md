# Neural Audio Watermarking Collusion Attack

针对现代神经音频水印（AudioSeal / WavMark / VoiceMark / WMCodec / TimbreWM）的多副本合谋攻击与篡改（framing）研究代码，对应论文的实验复现。

核心问题：经典线性平均（uniform averaging）在神经音频水印中何时仍是最优盲攻击，何时能被（a）盲波形几何修正或（b）payload-aware 合谋几何给出更强的凸攻击。

## 目录结构

```
.
├── src/                      # 核心库（本文贡献）
│   ├── watermarks.py         # 5 模型统一封装：embed / detect / 评估指标
│   ├── registry.py           # 全空间注册表 + 种子函数（payload 确定性复现）
│   └── convex.py             # 凸攻击权重求解（盲 + payload-aware 二次规划）
├── scripts/                  # 可执行实验脚本
│   ├── attack.py             # 攻击主表：mean / blind_gram_cb / extreme_pair
│   ├── framing.py            # 篡改：mean / framing_cb（payload-aware）
│   ├── blind_distance.py     # 盲方法：两两波形距离矩阵（精确变换）
│   ├── blind_minimax.py      # 盲方法：minimax 距离
│   ├── baselines.py          # 经典 baseline：median/min/max/rand_minmax/copy_paste
│   ├── minimax_framing.py    # minimax 变体（payload-aware 性质验证）
│   ├── pulse_noise.py        # 脉冲噪声对照（Kiyavash & Moulin）
│   ├── stats.py              # 按说话人分层的 McNemar + bootstrap
│   └── summary.py            # 汇总主表
├── third_party/              # 第三方模型封装代码（来自原仓库，各自许可证）
│   ├── voicemark/            # VoiceMark（含 SpeechTokenizer）
│   ├── wmcodec/              # WMCodec
│   └── timbrewm/             # TimbreWM（含 HiFi-GAN）
├── data/                     # 论文定稿数值快照（63 个 CSV + stats_all.txt）
├── tools/eval/               # 可选：独立音频质量评估工具（AudioEval，PESQ/STOI/SI-SDR/ViSQOL）
├── dataset/                  # 原始数据集放置处（libritts16k 等，见下）
├── DATA_INVENTORY.md         # 数据清单
└── LICENSE                   # MIT
```

## 安装

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch torchaudio  # 按 CUDA 版本安装
pip install -r requirements.txt
```

### 第三方模型权重（不随仓库分发，需自行下载）

本仓库只包含调用代码，**不含任何模型权重**。按下面放置权重文件：

| 模型 | 权重文件 → 放置位置 | 来源 |
|---|---|---|
| AudioSeal | 无需手动（`pip install audioseal` 后首次调用自动从 HuggingFace 下载）| [facebookresearch/audioseal](https://github.com/facebookresearch/audioseal)（MIT）|
| WavMark | 无需手动（`pip install wavmark` 权重内置）| [wavmark/wavmark](https://github.com/wavmark/wavmark)（MIT）|
| VoiceMark | `voicemark.pth` → `third_party/voicemark/`；`SpeechTokenizer.pt` → `third_party/voicemark/speechtokenizer/pretrained_model/` | VoiceMark 原仓库 + [SpeechTokenizer](https://github.com/ZhangXInFD/SpeechTokenizer)（Apache 2.0）|
| WMCodec | `g_00150000` → `third_party/wmcodec/save_model/` | WMCodec 原仓库 |
| TimbreWM | `compressed_none-conv2_ep_20_2023-01-17_23_01_01.pth.tar` → `third_party/timbrewm/results/ckpt/pth/` | [TimbreWatermarking](https://github.com/TimbreWatermarking/TimbreWatermarking) |

## 数据集

实验使用 libritts16k（LibriTTS 16kHz）。将原始 wav 放到 `dataset/libritts16k/`，命名约定为 `{spk}_{file}.wav`（见 `src/registry.py` 的 `clean_path_v19`）。

## 复现

```bash
# 攻击主表（150 trial × 3 方法）
python scripts/attack.py --model audioseal --K 5 --n_trials 150

# 篡改主表
python scripts/framing.py --model audioseal --K 5 --n_trials 150

# 统计检验（读 data/ 下已有 CSV）
python scripts/stats.py
```

每个脚本头部注释有完整参数说明。结果输出到 `results/evaluation/`（首次运行自动创建）。计算设备默认 `cuda:0`，可用环境变量 `WATERMARK_DEVICE` 覆盖（如 `WATERMARK_DEVICE=cpu python scripts/attack.py ...`）。

## 数据说明

`data/` 下是论文定稿数值快照，与 `DATA_INVENTORY.md` 一一对应。每个 CSV 的 trial 可确定性复现：给定 `(model, K, spk, local_t)`，`src/registry.py` 的 `coalition_seed` / `sample_coalition` / `int_to_bits` 可重算该 trial 的完整 payload 位向量，无需额外分发码本。

## 许可证

本项目代码 MIT License（见 `LICENSE`）。

`third_party/` 下代码来自第三方项目，保留各自许可证：
- SpeechTokenizer — Apache 2.0
- HiFi-GAN — MIT（`third_party/timbrewm/hifigan/LICENSE`）
- 其余模型封装代码遵循对应原仓库许可证，使用前请自行核对。

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
│   ├── attack.py             # 攻击主表：mean / fwp（farthest waveform pair）
│   ├── rp.py                 # RP（random pair）：均匀随机配对对照
│   ├── eep.py                # EEP（energy-extreme pair）：能量最高/最低配对对照
│   ├── framing.py            # 篡改：mean / tct（targeted convex tampering, payload-aware）
│   ├── blind_distance.py     # 盲方法：DM（dispersion maximization，两两波形距离矩阵精确变换）
│   ├── blind_minimax.py      # 盲方法：BDB（blind distance balancing）
│   ├── pgr.py                # PGR（payload geometry reference，payload-aware minimax 参照，评估用，非盲攻击）
│   ├── baselines.py          # 经典 baseline：median/min/max/rand_minmax/copy_paste
│   ├── minimax_framing.py    # minimax 变体性质验证（对称性/不规则几何，探针脚本，非正式数据产出）
│   ├── pulse_noise.py        # 脉冲噪声对照（Kiyavash & Moulin）
│   ├── stats.py              # 按说话人分层的 McNemar + bootstrap
│   └── summary.py            # 汇总主表
├── third_party/              # 第三方模型封装代码（来自原仓库，各自许可证）
│   ├── voicemark/            # VoiceMark（含 SpeechTokenizer）
│   ├── wmcodec/              # WMCodec
│   └── timbrewm/             # TimbreWM（含 HiFi-GAN）
├── data/                     # 唯一受 Git 跟踪的完整结果快照（按实验类别分目录）
├── results/                  # 本地运行时输出/checkpoint/log（不进入 Git）
├── tools/                    # 辅助工具
│   ├── eval/                 # 可选：独立音频质量评估（AudioEval，PESQ/STOI/SI-SDR/ViSQOL）
│   └── prepare_libritts16k.py  # 历史 LibriTTS 数据准备工具
├── dataset/                  # 本地音频数据集（不进入 Git）
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
| AudioSeal | 无需手动（`pip install audioseal` 后首次调用自动从 HuggingFace 下载）| [AudioSeal](https://arxiv.org/abs/2401.17264)（MIT）|
| WavMark | 无需手动（`pip install wavmark` 权重内置）| [WavMark](https://arxiv.org/abs/2308.12770)（MIT）|
| VoiceMark | `voicemark.pth` → `third_party/voicemark/`；`SpeechTokenizer.pt` → `third_party/voicemark/speechtokenizer/pretrained_model/` | [VoiceMark: Zero-Shot Voice Cloning-Resistant Watermarking Approach Leveraging Speaker-Specific Latents](https://arxiv.org/abs/2505.21568)（Interspeech 2025）|
| WMCodec | `g_00150000` → `third_party/wmcodec/save_model/` | [WMCodec: End-to-End Neural Speech Codec with Deep Watermarking for Authenticity Verification](https://arxiv.org/abs/2409.12121) |
| TimbreWM | `compressed_none-conv2_ep_20_2023-01-17_23_01_01.pth.tar` → `third_party/timbrewm/results/ckpt/pth/`；`generator_v1` → `third_party/timbrewm/hifigan/model/VCTK_V1/` | [官方代码与权重](https://github.com/TimbreWatermarking/TimbreWatermarking) |

## 数据集

正式实验使用本地 `dataset/collusion_300/`：

- English：LibriSpeech train-clean-100，50 位说话人；
- Chinese：AISHELL-3，50 位说话人；
- 每位说话人 3 条音频，共 300 条；
- 每条均为单声道 16 kHz、约 10 秒 PCM WAV；
- `manifest.csv` 保存 `language`、`speaker_id`、音频路径和来源信息。

`src/registry.py` 的 `speakers()` 从该 manifest 返回 `language:speaker_id` 格式的 100 个说话人；`coalition_seed()` 使用 SHA-256 确定性种子，`clean_path_v19()` 根据语言和说话人定位音频，`get_or_embed()` 使用带校验和原子写的并发安全缓存。旧版 `SPEAKERS_38` 和 `int(spk)` 哈希协议不再使用。

音频数据集、嵌入缓存和模型权重不会提交到 Git。`tools/prepare_libritts16k.py` 仅保留为历史数据准备工具，不对应当前 100 说话人主实验。

在此机器上使用已建好的虚拟环境：

```bash
source /private/users/lym/venv/bin/activate
export PYTHONPATH=src
export WATERMARK_DEVICE=cuda:0  # 可改为 cuda:1 至 cuda:6
```

## 方法命名

代码内部方法名（CSV `method` 列、函数名）与论文正文命名的对应关系：

| 代码内部名 | 论文命名 | 脚本 | 性质 |
|---|---|---|---|
| `mean` | Mean | `attack.py` | 盲攻击 |
| `fwp` | FWP（farthest waveform pair） | `attack.py` | 盲攻击 |
| `rp` | RP（random pair） | `rp.py` | 盲攻击对照 |
| `eep` | EEP（energy-extreme pair） | `eep.py` | 盲攻击对照 |
| `dm` | DM（dispersion maximization） | `blind_distance.py` | 盲攻击 |
| `bdb` | BDB（blind distance balancing） | `blind_minimax.py` | 盲攻击 |
| `pgr` | PGR（payload geometry reference） | `pgr.py` | **评估参照，非盲攻击**（用真实 payload Gram，只用于诊断"payload 知情能多赚多少"） |
| `tct` | TCT（targeted convex tampering） | `framing.py` | 篡改（payload-aware） |

`src/convex.py` 中的 `blind_gram_cb`（盲估计 Gram 的 CB 方法）**不在论文正文方法家族表中**，本轮不产出对应的 `data/` 数据，仅作为历史参照代码保留在 `src/convex.py`。

## 复现

每个实验脚本先将 CSV 和断点写入本地 `results/evaluation/`。该目录只用于运行时，不受 Git 跟踪；完整结果通过 `scripts/publish_results_to_data.py` 原子整理到受版本控制的 `data/`。计算设备默认 `cuda:0`，可用环境变量 `WATERMARK_DEVICE` 覆盖（如 `WATERMARK_DEVICE=cpu`）。

```bash
# 攻击主表（mean / fwp，300 trial）
python scripts/attack.py --model audioseal --K 5 --n_trials 300

# RP / EEP 对照（均匀随机配对 / 能量极值配对，300 trial）
python scripts/rp.py --model audioseal --K 5 --n_trials 300
python scripts/eep.py --model audioseal --K 5 --n_trials 300

# 经典 baseline（median/min/max/rand_minmax/copy_paste，300 trial）
python scripts/baselines.py --model audioseal --K 5 --n_trials 300

# 盲方法：DM（两两波形距离矩阵，精确变换）
python scripts/blind_distance.py --model audioseal --K 5 --n_trials 300

# 盲方法：BDB（blind distance balancing）
python scripts/blind_minimax.py --model audioseal --K 5 --n_trials 300

# PGR（payload-aware 参照，评估用，非盲攻击）
python scripts/pgr.py --model audioseal --K 5 --n_trials 300

# 篡改（payload-aware，mean / tct）
python scripts/framing.py --model audioseal --K 5 --n_trials 300

# 任意非合谋目标：每 trial 均匀采样 10 个 target
python scripts/framing.py --model audioseal --K 5 --n_trials 300 \
    --target_policy arbitrary

# matched-registry arbitrary tamper：完整 payload，不截位，只限制候选集 N=1024
python scripts/framing.py --model audioseal --K 5 --n_trials 300 \
    --target_policy arbitrary --registry_size 1024

# Registry-size control：mean/FWP；16-bit 模型含 N=256...65536 sweep
python scripts/registry_size_control.py --model audioseal --K 5 --n_trials 300

# 统一导出攻击后的音质与原生 presence
python scripts/export_quality_presence.py

# K=5 时移敏感性与独立 codec 敏感性
python scripts/temporal_sensitivity.py --model audioseal --n_trials 100
python scripts/codec_sensitivity.py --model audioseal --n_trials 300

# 脉冲噪声对照（Kiyavash & Moulin）
python scripts/pulse_noise.py --model audioseal --K 5 --n_trials 50

# 汇总主表（读 data/ 下已产出的 CSV）
python scripts/summary.py

# 统计检验（按说话人分层的 McNemar + bootstrap，读 data/ 下已有 CSV）
python scripts/stats.py --model audioseal --K 5

# 将所有已完成 CSV 分类覆盖到 data/，自动生成 README/INDEX/checksum
python scripts/publish_results_to_data.py
```

完整复现 = 5 个模型（`audioseal timbrewm wavmark voicemark wmcodec`）× 4 个 K（`2 3 5 8`）× 上述每个脚本。论文定稿的完整 CSV 快照已在 `data/` 下（见 `DATA_INVENTORY.md`），无需重跑即可复现表格与显著性检验（`summary.py` / `stats.py`）。

## 数据说明

`data/` 是唯一的发布结果目录，与 `DATA_INVENTORY.md` 一一对应；`data/INDEX.csv` 逐文件记录类别、行数、列名、大小、运行时来源路径和 SHA-256。`results/` 只保存本地 checkpoint/log，可随时由脚本继续运行，不应提交。

每个 trial 均可确定性复现：给定 `(model, K, spk, local_t)`，`src/registry.py` 的 `coalition_seed()`、`sample_coalition()` 和 `int_to_bits()` 可重建 coalition 与 payload。Matched-`N=1024` 实验从模型原生 registry 中独立抽取候选身份、强制包含所有 colluder，并保留完整的 10/16-bit payload；它不是 ECC，也没有截断 16-bit 模型的 6 位。

## 许可证

本项目代码 MIT License（见 `LICENSE`）。

`third_party/` 下代码来自第三方项目，保留各自许可证：
- SpeechTokenizer — Apache 2.0
- HiFi-GAN — MIT（`third_party/timbrewm/hifigan/LICENSE`）
- 其余模型封装代码遵循对应原仓库许可证，使用前请自行核对。

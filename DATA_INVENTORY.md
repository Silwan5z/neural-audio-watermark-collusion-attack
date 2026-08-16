# 论文数据清单

本清单列出本项目 `data/` 目录下当前实际存放的数据文件。相对路径以项目根目录（opensource/）为基准。后续增删数据时同步更新本清单。

## 攻击（全盲，5 模型 × K=2/3/5/8）

每个文件为 150 trial × 3 方法（mean / blind_gram_cb / extreme_pair）的全盲攻击结果，指标含 ASR、R3_escape、R5_escape、ACC_near、ACC_near_norm、AggResid 与音质 PESQ/STOI/SI-SDR。

| 相对路径 | 介绍 | 更改时间 |
|---|---|---|
| `data/attack/attack_audioseal_K2.csv` | audioseal 在 K=2 coalition 下的全盲攻击 | 2026-08-16 03:15:01 |
| `data/attack/attack_audioseal_K3.csv` | audioseal 在 K=3 coalition 下的全盲攻击 | 2026-08-16 03:17:18 |
| `data/attack/attack_audioseal_K5.csv` | audioseal 在 K=5 coalition 下的全盲攻击 | 2026-08-16 03:19:37 |
| `data/attack/attack_audioseal_K8.csv` | audioseal 在 K=8 coalition 下的全盲攻击 | 2026-08-16 03:21:58 |
| `data/attack/attack_timbrewm_K2.csv` | timbrewm 在 K=2 coalition 下的全盲攻击 | 2026-08-16 03:05:38 |
| `data/attack/attack_timbrewm_K3.csv` | timbrewm 在 K=3 coalition 下的全盲攻击 | 2026-08-16 03:07:58 |
| `data/attack/attack_timbrewm_K5.csv` | timbrewm 在 K=5 coalition 下的全盲攻击 | 2026-08-16 03:10:21 |
| `data/attack/attack_timbrewm_K8.csv` | timbrewm 在 K=8 coalition 下的全盲攻击 | 2026-08-16 03:12:44 |
| `data/attack/attack_voicemark_K2.csv` | voicemark 在 K=2 coalition 下的全盲攻击 | 2026-08-16 02:44:59 |
| `data/attack/attack_voicemark_K3.csv` | voicemark 在 K=3 coalition 下的全盲攻击 | 2026-08-16 02:47:39 |
| `data/attack/attack_voicemark_K5.csv` | voicemark 在 K=5 coalition 下的全盲攻击 | 2026-08-16 02:50:24 |
| `data/attack/attack_voicemark_K8.csv` | voicemark 在 K=8 coalition 下的全盲攻击 | 2026-08-16 02:53:09 |
| `data/attack/attack_wavmark_K2.csv` | wavmark 在 K=2 coalition 下的全盲攻击 | 2026-08-16 03:39:17 |
| `data/attack/attack_wavmark_K3.csv` | wavmark 在 K=3 coalition 下的全盲攻击 | 2026-08-16 04:25:25 |
| `data/attack/attack_wavmark_K5.csv` | wavmark 在 K=5 coalition 下的全盲攻击 | 2026-08-16 05:11:35 |
| `data/attack/attack_wavmark_K8.csv` | wavmark 在 K=8 coalition 下的全盲攻击 | 2026-08-16 05:57:50 |
| `data/attack/attack_wmcodec_K2.csv` | wmcodec 在 K=2 coalition 下的全盲攻击 | 2026-08-16 02:55:40 |
| `data/attack/attack_wmcodec_K3.csv` | wmcodec 在 K=3 coalition 下的全盲攻击 | 2026-08-16 02:58:13 |
| `data/attack/attack_wmcodec_K5.csv` | wmcodec 在 K=5 coalition 下的全盲攻击 | 2026-08-16 03:00:44 |
| `data/attack/attack_wmcodec_K8.csv` | wmcodec 在 K=8 coalition 下的全盲攻击 | 2026-08-16 03:03:20 |

## 盲方法 blind_dist_cb（全盲，5 模型 × K=2/3/5/8）

每个文件为 150 trial × 单方法（blind_dist_cb，盲距离变换（两两波形距离矩阵的精确变换））的全盲攻击结果，指标含 ASR、R3_escape、R5_escape、ACC_near、ACC_near_norm 与音质 PESQ/STOI/SI-SDR。

| 相对路径 | 介绍 | 更改时间 |
|---|---|---|
| `data/blind_dist_cb/blind_dist_cb_audioseal_K2.csv` | audioseal 在 K=2 coalition 下的盲距离变换攻击 | 2026-08-16 18:12:16 |
| `data/blind_dist_cb/blind_dist_cb_audioseal_K3.csv` | audioseal 在 K=3 coalition 下的盲距离变换攻击 | 2026-08-16 18:13:11 |
| `data/blind_dist_cb/blind_dist_cb_audioseal_K5.csv` | audioseal 在 K=5 coalition 下的盲距离变换攻击 | 2026-08-16 18:14:10 |
| `data/blind_dist_cb/blind_dist_cb_audioseal_K8.csv` | audioseal 在 K=8 coalition 下的盲距离变换攻击 | 2026-08-16 18:15:20 |
| `data/blind_dist_cb/blind_dist_cb_timbrewm_K2.csv` | timbrewm 在 K=2 coalition 下的盲距离变换攻击 | 2026-08-16 18:08:18 |
| `data/blind_dist_cb/blind_dist_cb_timbrewm_K3.csv` | timbrewm 在 K=3 coalition 下的盲距离变换攻击 | 2026-08-16 18:09:14 |
| `data/blind_dist_cb/blind_dist_cb_timbrewm_K5.csv` | timbrewm 在 K=5 coalition 下的盲距离变换攻击 | 2026-08-16 18:10:15 |
| `data/blind_dist_cb/blind_dist_cb_timbrewm_K8.csv` | timbrewm 在 K=8 coalition 下的盲距离变换攻击 | 2026-08-16 18:11:27 |
| `data/blind_dist_cb/blind_dist_cb_wavmark_K2.csv` | wavmark 在 K=2 coalition 下的盲距离变换攻击 | 2026-08-16 18:21:08 |
| `data/blind_dist_cb/blind_dist_cb_wavmark_K3.csv` | wavmark 在 K=3 coalition 下的盲距离变换攻击 | 2026-08-16 18:27:01 |
| `data/blind_dist_cb/blind_dist_cb_wavmark_K5.csv` | wavmark 在 K=5 coalition 下的盲距离变换攻击 | 2026-08-16 18:32:57 |
| `data/blind_dist_cb/blind_dist_cb_wavmark_K8.csv` | wavmark 在 K=8 coalition 下的盲距离变换攻击 | 2026-08-16 18:38:56 |
| `data/blind_dist_cb/blind_dist_cb_voicemark_K2.csv` | voicemark 在 K=2 coalition 下的盲距离变换攻击 | 2026-08-16 17:59:38 |
| `data/blind_dist_cb/blind_dist_cb_voicemark_K3.csv` | voicemark 在 K=3 coalition 下的盲距离变换攻击 | 2026-08-16 18:00:39 |
| `data/blind_dist_cb/blind_dist_cb_voicemark_K5.csv` | voicemark 在 K=5 coalition 下的盲距离变换攻击 | 2026-08-16 18:01:50 |
| `data/blind_dist_cb/blind_dist_cb_voicemark_K8.csv` | voicemark 在 K=8 coalition 下的盲距离变换攻击 | 2026-08-16 18:03:11 |
| `data/blind_dist_cb/blind_dist_cb_wmcodec_K2.csv` | wmcodec 在 K=2 coalition 下的盲距离变换攻击 | 2026-08-16 18:04:08 |
| `data/blind_dist_cb/blind_dist_cb_wmcodec_K3.csv` | wmcodec 在 K=3 coalition 下的盲距离变换攻击 | 2026-08-16 18:05:08 |
| `data/blind_dist_cb/blind_dist_cb_wmcodec_K5.csv` | wmcodec 在 K=5 coalition 下的盲距离变换攻击 | 2026-08-16 18:06:14 |
| `data/blind_dist_cb/blind_dist_cb_wmcodec_K8.csv` | wmcodec 在 K=8 coalition 下的盲距离变换攻击 | 2026-08-16 18:07:27 |

## 盲方法 blind_minimax_cb（全盲，5 模型 × K=2/3/5/8）

每个文件为 150 trial × 单方法（blind_minimax_cb，盲 minimax 距离）的全盲攻击结果，指标含 ASR、R3_escape、R5_escape、ACC_near、ACC_near_norm 与音质 PESQ/STOI/SI-SDR。

| 相对路径 | 介绍 | 更改时间 |
|---|---|---|
| `data/blind_minimax_cb/blind_minimax_cb_audioseal_K2.csv` | audioseal 在 K=2 coalition 下的盲 minimax 距离攻击 | 2026-08-16 18:52:21 |
| `data/blind_minimax_cb/blind_minimax_cb_audioseal_K3.csv` | audioseal 在 K=3 coalition 下的盲 minimax 距离攻击 | 2026-08-16 18:53:12 |
| `data/blind_minimax_cb/blind_minimax_cb_audioseal_K5.csv` | audioseal 在 K=5 coalition 下的盲 minimax 距离攻击 | 2026-08-16 18:54:11 |
| `data/blind_minimax_cb/blind_minimax_cb_audioseal_K8.csv` | audioseal 在 K=8 coalition 下的盲 minimax 距离攻击 | 2026-08-16 18:55:34 |
| `data/blind_minimax_cb/blind_minimax_cb_timbrewm_K2.csv` | timbrewm 在 K=2 coalition 下的盲 minimax 距离攻击 | 2026-08-16 18:48:17 |
| `data/blind_minimax_cb/blind_minimax_cb_timbrewm_K3.csv` | timbrewm 在 K=3 coalition 下的盲 minimax 距离攻击 | 2026-08-16 18:49:10 |
| `data/blind_minimax_cb/blind_minimax_cb_timbrewm_K5.csv` | timbrewm 在 K=5 coalition 下的盲 minimax 距离攻击 | 2026-08-16 18:50:11 |
| `data/blind_minimax_cb/blind_minimax_cb_timbrewm_K8.csv` | timbrewm 在 K=8 coalition 下的盲 minimax 距离攻击 | 2026-08-16 18:51:32 |
| `data/blind_minimax_cb/blind_minimax_cb_wavmark_K2.csv` | wavmark 在 K=2 coalition 下的盲 minimax 距离攻击 | 2026-08-16 19:01:22 |
| `data/blind_minimax_cb/blind_minimax_cb_wavmark_K3.csv` | wavmark 在 K=3 coalition 下的盲 minimax 距离攻击 | 2026-08-16 19:07:11 |
| `data/blind_minimax_cb/blind_minimax_cb_wavmark_K5.csv` | wavmark 在 K=5 coalition 下的盲 minimax 距离攻击 | 2026-08-16 19:13:01 |
| `data/blind_minimax_cb/blind_minimax_cb_wavmark_K8.csv` | wavmark 在 K=8 coalition 下的盲 minimax 距离攻击 | 2026-08-16 19:18:56 |
| `data/blind_minimax_cb/blind_minimax_cb_voicemark_K2.csv` | voicemark 在 K=2 coalition 下的盲 minimax 距离攻击 | 2026-08-16 18:39:53 |
| `data/blind_minimax_cb/blind_minimax_cb_voicemark_K3.csv` | voicemark 在 K=3 coalition 下的盲 minimax 距离攻击 | 2026-08-16 18:40:53 |
| `data/blind_minimax_cb/blind_minimax_cb_voicemark_K5.csv` | voicemark 在 K=5 coalition 下的盲 minimax 距离攻击 | 2026-08-16 18:42:02 |
| `data/blind_minimax_cb/blind_minimax_cb_voicemark_K8.csv` | voicemark 在 K=8 coalition 下的盲 minimax 距离攻击 | 2026-08-16 18:43:14 |
| `data/blind_minimax_cb/blind_minimax_cb_wmcodec_K2.csv` | wmcodec 在 K=2 coalition 下的盲 minimax 距离攻击 | 2026-08-16 18:44:10 |
| `data/blind_minimax_cb/blind_minimax_cb_wmcodec_K3.csv` | wmcodec 在 K=3 coalition 下的盲 minimax 距离攻击 | 2026-08-16 18:45:09 |
| `data/blind_minimax_cb/blind_minimax_cb_wmcodec_K5.csv` | wmcodec 在 K=5 coalition 下的盲 minimax 距离攻击 | 2026-08-16 18:46:15 |
| `data/blind_minimax_cb/blind_minimax_cb_wmcodec_K8.csv` | wmcodec 在 K=8 coalition 下的盲 minimax 距离攻击 | 2026-08-16 18:47:27 |

## 经典 baseline（全盲，5 模型 × K=2/3/5/8）

每个文件为 150 trial × 5 方法（median / minimum / maximum / rand_minmax / copy_paste）的经典 baseline 结果，指标含 ASR、R3_escape、R5_escape、ACC_near、ACC_near_norm 与音质 PESQ/STOI/SI-SDR。

| 相对路径 | 介绍 | 更改时间 |
|---|---|---|
| `data/baselines/baselines_audioseal_K2.csv` | audioseal 在 K=2 coalition 下的经典 baseline | 2026-08-16 02:02:26 |
| `data/baselines/baselines_audioseal_K3.csv` | audioseal 在 K=3 coalition 下的经典 baseline | 2026-08-16 02:08:26 |
| `data/baselines/baselines_audioseal_K5.csv` | audioseal 在 K=5 coalition 下的经典 baseline | 2026-08-16 02:14:31 |
| `data/baselines/baselines_audioseal_K8.csv` | audioseal 在 K=8 coalition 下的经典 baseline | 2026-08-16 01:16:03 |
| `data/baselines/baselines_timbrewm_K2.csv` | timbrewm 在 K=2 coalition 下的经典 baseline | 2026-08-16 01:32:39 |
| `data/baselines/baselines_timbrewm_K3.csv` | timbrewm 在 K=3 coalition 下的经典 baseline | 2026-08-16 01:44:06 |
| `data/baselines/baselines_timbrewm_K5.csv` | timbrewm 在 K=5 coalition 下的经典 baseline | 2026-08-16 01:50:10 |
| `data/baselines/baselines_timbrewm_K8.csv` | timbrewm 在 K=8 coalition 下的经典 baseline | 2026-08-16 01:56:22 |
| `data/baselines/baselines_voicemark_K2.csv` | voicemark 在 K=2 coalition 下的经典 baseline | 2026-08-16 01:14:31 |
| `data/baselines/baselines_voicemark_K3.csv` | voicemark 在 K=3 coalition 下的经典 baseline | 2026-08-16 01:15:18 |
| `data/baselines/baselines_voicemark_K5.csv` | voicemark 在 K=5 coalition 下的经典 baseline | 2026-08-16 01:32:55 |
| `data/baselines/baselines_voicemark_K8.csv` | voicemark 在 K=8 coalition 下的经典 baseline | 2026-08-16 01:15:44 |
| `data/baselines/baselines_wavmark_K2.csv` | wavmark 在 K=2 coalition 下的经典 baseline | 2026-08-16 04:08:06 |
| `data/baselines/baselines_wavmark_K3.csv` | wavmark 在 K=3 coalition 下的经典 baseline | 2026-08-16 04:54:14 |
| `data/baselines/baselines_wavmark_K5.csv` | wavmark 在 K=5 coalition 下的经典 baseline | 2026-08-16 05:40:25 |
| `data/baselines/baselines_wavmark_K8.csv` | wavmark 在 K=8 coalition 下的经典 baseline | 2026-08-16 06:26:41 |
| `data/baselines/baselines_wmcodec_K2.csv` | wmcodec 在 K=2 coalition 下的经典 baseline | 2026-08-16 01:15:24 |
| `data/baselines/baselines_wmcodec_K3.csv` | wmcodec 在 K=3 coalition 下的经典 baseline | 2026-08-16 01:32:07 |
| `data/baselines/baselines_wmcodec_K5.csv` | wmcodec 在 K=5 coalition 下的经典 baseline | 2026-08-16 01:32:40 |
| `data/baselines/baselines_wmcodec_K8.csv` | wmcodec 在 K=8 coalition 下的经典 baseline | 2026-08-16 01:33:05 |

## 篡改（payload-aware，5 模型 × K=2/3/5/8）

每个文件为 150 trial × 10 候选 target × 2 方法（mean / framing_cb）的 payload-aware 篡改结果，指标为 target_top1。

| 相对路径 | 介绍 | 更改时间 |
|---|---|---|
| `data/tamper/tamper_audioseal_K2.csv` | audioseal 在 K=2 coalition 下的 payload-aware 篡改 | 2026-08-16 09:18:13 |
| `data/tamper/tamper_audioseal_K3.csv` | audioseal 在 K=3 coalition 下的 payload-aware 篡改 | 2026-08-16 09:28:08 |
| `data/tamper/tamper_audioseal_K5.csv` | audioseal 在 K=5 coalition 下的 payload-aware 篡改 | 2026-08-16 09:47:08 |
| `data/tamper/tamper_audioseal_K8.csv` | audioseal 在 K=8 coalition 下的 payload-aware 篡改 | 2026-08-16 10:14:15 |
| `data/tamper/tamper_timbrewm_K2.csv` | timbrewm 在 K=2 coalition 下的 payload-aware 篡改 | 2026-08-16 08:46:23 |
| `data/tamper/tamper_timbrewm_K3.csv` | timbrewm 在 K=3 coalition 下的 payload-aware 篡改 | 2026-08-16 08:51:37 |
| `data/tamper/tamper_timbrewm_K5.csv` | timbrewm 在 K=5 coalition 下的 payload-aware 篡改 | 2026-08-16 09:00:31 |
| `data/tamper/tamper_timbrewm_K8.csv` | timbrewm 在 K=8 coalition 下的 payload-aware 篡改 | 2026-08-16 09:12:57 |
| `data/tamper/tamper_voicemark_K2.csv` | voicemark 在 K=2 coalition 下的 payload-aware 篡改 | 2026-08-16 06:34:10 |
| `data/tamper/tamper_voicemark_K3.csv` | voicemark 在 K=3 coalition 下的 payload-aware 篡改 | 2026-08-16 06:46:21 |
| `data/tamper/tamper_voicemark_K5.csv` | voicemark 在 K=5 coalition 下的 payload-aware 篡改 | 2026-08-16 07:07:59 |
| `data/tamper/tamper_voicemark_K8.csv` | voicemark 在 K=8 coalition 下的 payload-aware 篡改 | 2026-08-16 07:38:26 |
| `data/tamper/tamper_wavmark_K2.csv` | wavmark 在 K=2 coalition 下的 payload-aware 篡改 | 2026-08-16 12:00:10 |
| `data/tamper/tamper_wavmark_K3.csv` | wavmark 在 K=3 coalition 下的 payload-aware 篡改 | 2026-08-16 13:50:26 |
| `data/tamper/tamper_wavmark_K5.csv` | wavmark 在 K=5 coalition 下的 payload-aware 篡改 | 2026-08-16 15:49:30 |
| `data/tamper/tamper_wavmark_K8.csv` | wavmark 在 K=8 coalition 下的 payload-aware 篡改 | 2026-08-16 17:57:31 |
| `data/tamper/tamper_wmcodec_K2.csv` | wmcodec 在 K=2 coalition 下的 payload-aware 篡改 | 2026-08-16 07:44:34 |
| `data/tamper/tamper_wmcodec_K3.csv` | wmcodec 在 K=3 coalition 下的 payload-aware 篡改 | 2026-08-16 07:55:12 |
| `data/tamper/tamper_wmcodec_K5.csv` | wmcodec 在 K=5 coalition 下的 payload-aware 篡改 | 2026-08-16 08:15:03 |
| `data/tamper/tamper_wmcodec_K8.csv` | wmcodec 在 K=8 coalition 下的 payload-aware 篡改 | 2026-08-16 08:43:11 |

## 脉冲噪声对照

每个文件为按说话人聚合的 ASR 扫描，12 个 (eps, r0) 组合（eps ∈ {0, 0.001, 0.01, 0.05}，r0 ∈ {0.05, 0.1, 0.3}），38 说话人。

| 相对路径 | 介绍 | 更改时间 |
|---|---|---|
| `data/pulse_noise/pulse_noise_timbrewm_K5.csv` | timbrewm 在 K=5 下的脉冲噪声 ASR 扫描（mean 凸混合 + 两点分布噪声）| 2026-08-16 17:57:50 |
| `data/pulse_noise/pulse_noise_audioseal_K5.csv` | audioseal 在 K=5 下的脉冲噪声 ASR 扫描（mean 凸混合 + 两点分布噪声）| 2026-08-16 17:58:05 |

## 统计检验

| 相对路径 | 介绍 | 更改时间 |
|---|---|---|
| `data/stats_all.txt` | 按说话人分层的 McNemar 检验 + paired bootstrap 置信区间，覆盖 5 模型 × 4K × ASR/ACC_near 两口径 | 2026-08-16 17:48:18 |

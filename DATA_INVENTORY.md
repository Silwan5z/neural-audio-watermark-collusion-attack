# 论文数据清单

本清单列出本项目 `data/` 目录下当前实际存放的数据文件。相对路径以项目根目录（opensource/）为基准。后续增删数据时同步更新本清单。

方法命名对照见 `README.md` 的“方法命名”一节（代码内部名 ↔ 论文命名）。

## 攻击（全盲，5 模型 × K=2/3/5/8）

每个文件为 300 trial × 2 方法（mean / fwp）的全盲攻击结果，指标含 ASR、R3_escape、R5_escape、ACC_near、ACC_near_norm、AggResid 与音质 PESQ/STOI/SI-SDR。

原 `blind_gram_cb`（盲估计 Gram 的 CB 方法）不在论文正文方法家族表中，本轮不再产出其数据（代码仍保留在 `src/convex.py` 作历史参照）。

| 相对路径 | 介绍 | 更改时间 |
|---|---|---|
| `data/attack/attack_audioseal_K2.csv` | audioseal 在 K=2 coalition 下的全盲攻击（mean/fwp） | 2026-08-17 17:48:25 |
| `data/attack/attack_audioseal_K3.csv` | audioseal 在 K=3 coalition 下的全盲攻击（mean/fwp） | 2026-08-17 17:48:25 |
| `data/attack/attack_audioseal_K5.csv` | audioseal 在 K=5 coalition 下的全盲攻击（mean/fwp） | 2026-08-17 17:48:25 |
| `data/attack/attack_audioseal_K8.csv` | audioseal 在 K=8 coalition 下的全盲攻击（mean/fwp） | 2026-08-17 17:48:25 |
| `data/attack/attack_timbrewm_K2.csv` | timbrewm 在 K=2 coalition 下的全盲攻击（mean/fwp） | 2026-08-17 17:48:25 |
| `data/attack/attack_timbrewm_K3.csv` | timbrewm 在 K=3 coalition 下的全盲攻击（mean/fwp） | 2026-08-17 17:48:25 |
| `data/attack/attack_timbrewm_K5.csv` | timbrewm 在 K=5 coalition 下的全盲攻击（mean/fwp） | 2026-08-17 17:48:25 |
| `data/attack/attack_timbrewm_K8.csv` | timbrewm 在 K=8 coalition 下的全盲攻击（mean/fwp） | 2026-08-17 17:48:25 |
| `data/attack/attack_voicemark_K2.csv` | voicemark 在 K=2 coalition 下的全盲攻击（mean/fwp） | 2026-08-17 17:48:25 |
| `data/attack/attack_voicemark_K3.csv` | voicemark 在 K=3 coalition 下的全盲攻击（mean/fwp） | 2026-08-17 17:48:25 |
| `data/attack/attack_voicemark_K5.csv` | voicemark 在 K=5 coalition 下的全盲攻击（mean/fwp） | 2026-08-17 17:48:25 |
| `data/attack/attack_voicemark_K8.csv` | voicemark 在 K=8 coalition 下的全盲攻击（mean/fwp） | 2026-08-17 17:48:26 |
| `data/attack/attack_wavmark_K2.csv` | wavmark 在 K=2 coalition 下的全盲攻击（mean/fwp） | 2026-08-17 17:48:25 |
| `data/attack/attack_wavmark_K3.csv` | wavmark 在 K=3 coalition 下的全盲攻击（mean/fwp） | 2026-08-17 17:48:25 |
| `data/attack/attack_wavmark_K5.csv` | wavmark 在 K=5 coalition 下的全盲攻击（mean/fwp） | 2026-08-17 17:48:25 |
| `data/attack/attack_wavmark_K8.csv` | wavmark 在 K=8 coalition 下的全盲攻击（mean/fwp） | 2026-08-17 17:48:25 |
| `data/attack/attack_wmcodec_K2.csv` | wmcodec 在 K=2 coalition 下的全盲攻击（mean/fwp） | 2026-08-17 17:48:26 |
| `data/attack/attack_wmcodec_K3.csv` | wmcodec 在 K=3 coalition 下的全盲攻击（mean/fwp） | 2026-08-17 17:48:26 |
| `data/attack/attack_wmcodec_K5.csv` | wmcodec 在 K=5 coalition 下的全盲攻击（mean/fwp） | 2026-08-17 17:48:26 |
| `data/attack/attack_wmcodec_K8.csv` | wmcodec 在 K=8 coalition 下的全盲攻击（mean/fwp） | 2026-08-17 17:48:26 |

## 盲方法 DM（Dispersion Maximization，全盲，5 模型 × K=2/3/5/8）

每个文件为 300 trial × 单方法（dm，`max_a a^T D a`）的全盲攻击结果，指标含 ASR、R3_escape、R5_escape、ACC_near、ACC_near_norm 与音质 PESQ/STOI/SI-SDR。原目录名 `blind_dist_cb` 已改为 `dm`。

| 相对路径 | 介绍 | 更改时间 |
|---|---|---|
| `data/dm/dm_audioseal_K2.csv` | audioseal 在 K=2 coalition 下的 DM 攻击 | 2026-08-17 17:48:25 |
| `data/dm/dm_audioseal_K3.csv` | audioseal 在 K=3 coalition 下的 DM 攻击 | 2026-08-17 17:48:25 |
| `data/dm/dm_audioseal_K5.csv` | audioseal 在 K=5 coalition 下的 DM 攻击 | 2026-08-17 17:48:25 |
| `data/dm/dm_audioseal_K8.csv` | audioseal 在 K=8 coalition 下的 DM 攻击 | 2026-08-17 17:48:25 |
| `data/dm/dm_timbrewm_K2.csv` | timbrewm 在 K=2 coalition 下的 DM 攻击 | 2026-08-17 17:48:25 |
| `data/dm/dm_timbrewm_K3.csv` | timbrewm 在 K=3 coalition 下的 DM 攻击 | 2026-08-17 17:48:25 |
| `data/dm/dm_timbrewm_K5.csv` | timbrewm 在 K=5 coalition 下的 DM 攻击 | 2026-08-17 17:48:25 |
| `data/dm/dm_timbrewm_K8.csv` | timbrewm 在 K=8 coalition 下的 DM 攻击 | 2026-08-17 17:48:25 |
| `data/dm/dm_voicemark_K2.csv` | voicemark 在 K=2 coalition 下的 DM 攻击 | 2026-08-17 17:48:25 |
| `data/dm/dm_voicemark_K3.csv` | voicemark 在 K=3 coalition 下的 DM 攻击 | 2026-08-17 17:48:25 |
| `data/dm/dm_voicemark_K5.csv` | voicemark 在 K=5 coalition 下的 DM 攻击 | 2026-08-17 17:48:25 |
| `data/dm/dm_voicemark_K8.csv` | voicemark 在 K=8 coalition 下的 DM 攻击 | 2026-08-17 17:48:26 |
| `data/dm/dm_wavmark_K2.csv` | wavmark 在 K=2 coalition 下的 DM 攻击 | 2026-08-17 17:48:25 |
| `data/dm/dm_wavmark_K3.csv` | wavmark 在 K=3 coalition 下的 DM 攻击 | 2026-08-17 17:48:25 |
| `data/dm/dm_wavmark_K5.csv` | wavmark 在 K=5 coalition 下的 DM 攻击 | 2026-08-17 17:48:25 |
| `data/dm/dm_wavmark_K8.csv` | wavmark 在 K=8 coalition 下的 DM 攻击 | 2026-08-17 17:48:25 |
| `data/dm/dm_wmcodec_K2.csv` | wmcodec 在 K=2 coalition 下的 DM 攻击 | 2026-08-17 17:48:26 |
| `data/dm/dm_wmcodec_K3.csv` | wmcodec 在 K=3 coalition 下的 DM 攻击 | 2026-08-17 17:48:26 |
| `data/dm/dm_wmcodec_K5.csv` | wmcodec 在 K=5 coalition 下的 DM 攻击 | 2026-08-17 17:48:26 |
| `data/dm/dm_wmcodec_K8.csv` | wmcodec 在 K=8 coalition 下的 DM 攻击 | 2026-08-17 17:48:26 |

## 盲方法 BDB（Blind Distance Balancing，全盲，5 模型 × K=2/3/5/8）

每个文件为 300 trial × 单方法（bdb，`max_a min_i[Da]_i`）的全盲攻击结果，指标含 ASR、R3_escape、R5_escape、ACC_near、ACC_near_norm 与音质 PESQ/STOI/SI-SDR。原目录名 `blind_minimax_cb` 已改为 `bdb`。

| 相对路径 | 介绍 | 更改时间 |
|---|---|---|
| `data/bdb/bdb_audioseal_K2.csv` | audioseal 在 K=2 coalition 下的 BDB 攻击 | 2026-08-17 17:48:25 |
| `data/bdb/bdb_audioseal_K3.csv` | audioseal 在 K=3 coalition 下的 BDB 攻击 | 2026-08-17 17:48:25 |
| `data/bdb/bdb_audioseal_K5.csv` | audioseal 在 K=5 coalition 下的 BDB 攻击 | 2026-08-17 17:48:25 |
| `data/bdb/bdb_audioseal_K8.csv` | audioseal 在 K=8 coalition 下的 BDB 攻击 | 2026-08-17 17:48:25 |
| `data/bdb/bdb_timbrewm_K2.csv` | timbrewm 在 K=2 coalition 下的 BDB 攻击 | 2026-08-17 17:48:25 |
| `data/bdb/bdb_timbrewm_K3.csv` | timbrewm 在 K=3 coalition 下的 BDB 攻击 | 2026-08-17 17:48:25 |
| `data/bdb/bdb_timbrewm_K5.csv` | timbrewm 在 K=5 coalition 下的 BDB 攻击 | 2026-08-17 17:48:25 |
| `data/bdb/bdb_timbrewm_K8.csv` | timbrewm 在 K=8 coalition 下的 BDB 攻击 | 2026-08-17 17:48:25 |
| `data/bdb/bdb_voicemark_K2.csv` | voicemark 在 K=2 coalition 下的 BDB 攻击 | 2026-08-17 17:48:25 |
| `data/bdb/bdb_voicemark_K3.csv` | voicemark 在 K=3 coalition 下的 BDB 攻击 | 2026-08-17 17:48:25 |
| `data/bdb/bdb_voicemark_K5.csv` | voicemark 在 K=5 coalition 下的 BDB 攻击 | 2026-08-17 17:48:25 |
| `data/bdb/bdb_voicemark_K8.csv` | voicemark 在 K=8 coalition 下的 BDB 攻击 | 2026-08-17 17:48:26 |
| `data/bdb/bdb_wavmark_K2.csv` | wavmark 在 K=2 coalition 下的 BDB 攻击 | 2026-08-17 17:48:25 |
| `data/bdb/bdb_wavmark_K3.csv` | wavmark 在 K=3 coalition 下的 BDB 攻击 | 2026-08-17 17:48:25 |
| `data/bdb/bdb_wavmark_K5.csv` | wavmark 在 K=5 coalition 下的 BDB 攻击 | 2026-08-17 17:48:25 |
| `data/bdb/bdb_wavmark_K8.csv` | wavmark 在 K=8 coalition 下的 BDB 攻击 | 2026-08-17 17:48:25 |
| `data/bdb/bdb_wmcodec_K2.csv` | wmcodec 在 K=2 coalition 下的 BDB 攻击 | 2026-08-17 17:48:26 |
| `data/bdb/bdb_wmcodec_K3.csv` | wmcodec 在 K=3 coalition 下的 BDB 攻击 | 2026-08-17 17:48:26 |
| `data/bdb/bdb_wmcodec_K5.csv` | wmcodec 在 K=5 coalition 下的 BDB 攻击 | 2026-08-17 17:48:26 |
| `data/bdb/bdb_wmcodec_K8.csv` | wmcodec 在 K=8 coalition 下的 BDB 攻击 | 2026-08-17 17:48:26 |

## PGR（Payload Geometry Reference，评估参照，非盲，5 模型 × K=2/3/5/8）

每个文件为 300 trial × 单方法（pgr，`min_a max_i[G_c a]_i`，使用真实 payload codeword Gram G_c）的结果，指标含 ASR、R3_escape、R5_escape、ACC_near、AggResid 与音质 PESQ/STOI/SI-SDR。**PGR 不是盲攻击**，仅用于诊断"payload 知情能比盲攻击多赚多少"，任何盲方法都不会读取此数据构造攻击向量。新增目录（v9 命名迁移前不存在）。

| 相对路径 | 介绍 | 更改时间 |
|---|---|---|
| `data/pgr/pgr_audioseal_K2.csv` | audioseal 在 K=2 coalition 下的 PGR 参照 | 2026-08-17 17:48:25 |
| `data/pgr/pgr_audioseal_K3.csv` | audioseal 在 K=3 coalition 下的 PGR 参照 | 2026-08-17 17:48:25 |
| `data/pgr/pgr_audioseal_K5.csv` | audioseal 在 K=5 coalition 下的 PGR 参照 | 2026-08-17 17:48:25 |
| `data/pgr/pgr_audioseal_K8.csv` | audioseal 在 K=8 coalition 下的 PGR 参照 | 2026-08-17 17:48:25 |
| `data/pgr/pgr_timbrewm_K2.csv` | timbrewm 在 K=2 coalition 下的 PGR 参照 | 2026-08-17 17:48:25 |
| `data/pgr/pgr_timbrewm_K3.csv` | timbrewm 在 K=3 coalition 下的 PGR 参照 | 2026-08-17 17:48:25 |
| `data/pgr/pgr_timbrewm_K5.csv` | timbrewm 在 K=5 coalition 下的 PGR 参照 | 2026-08-17 17:48:25 |
| `data/pgr/pgr_timbrewm_K8.csv` | timbrewm 在 K=8 coalition 下的 PGR 参照 | 2026-08-17 17:48:25 |
| `data/pgr/pgr_voicemark_K2.csv` | voicemark 在 K=2 coalition 下的 PGR 参照 | 2026-08-17 17:48:25 |
| `data/pgr/pgr_voicemark_K3.csv` | voicemark 在 K=3 coalition 下的 PGR 参照 | 2026-08-17 17:48:25 |
| `data/pgr/pgr_voicemark_K5.csv` | voicemark 在 K=5 coalition 下的 PGR 参照 | 2026-08-17 17:48:25 |
| `data/pgr/pgr_voicemark_K8.csv` | voicemark 在 K=8 coalition 下的 PGR 参照 | 2026-08-17 17:48:26 |
| `data/pgr/pgr_wavmark_K2.csv` | wavmark 在 K=2 coalition 下的 PGR 参照 | 2026-08-17 17:48:25 |
| `data/pgr/pgr_wavmark_K3.csv` | wavmark 在 K=3 coalition 下的 PGR 参照 | 2026-08-17 17:48:25 |
| `data/pgr/pgr_wavmark_K5.csv` | wavmark 在 K=5 coalition 下的 PGR 参照 | 2026-08-17 17:48:25 |
| `data/pgr/pgr_wavmark_K8.csv` | wavmark 在 K=8 coalition 下的 PGR 参照 | 2026-08-17 17:48:25 |
| `data/pgr/pgr_wmcodec_K2.csv` | wmcodec 在 K=2 coalition 下的 PGR 参照 | 2026-08-17 17:48:26 |
| `data/pgr/pgr_wmcodec_K3.csv` | wmcodec 在 K=3 coalition 下的 PGR 参照 | 2026-08-17 17:48:26 |
| `data/pgr/pgr_wmcodec_K5.csv` | wmcodec 在 K=5 coalition 下的 PGR 参照 | 2026-08-17 17:48:26 |
| `data/pgr/pgr_wmcodec_K8.csv` | wmcodec 在 K=8 coalition 下的 PGR 参照 | 2026-08-17 17:48:26 |

## RP / EEP（盲攻击对照基线，5 模型 × K=2/3/5/8）— 数据生成中，尚未落地

`rp`（random pair，随机取两路凸组合）与 `eep`（energy-extreme pair，取能量最高/最低两路凸组合）是新增的盲攻击对照基线，脚本为 `scripts/rp.py` / `scripts/eep.py`。全量 n=300 数据正在后台生成（40 组 model×K 运行，排在其他 GPU 任务之后），**本次同步尚未包含 `data/rp/` 与 `data/eep/`**。数据生成完成后需补充本节并放入对应目录。

## 经典 baseline（全盲，5 模型 × K=2/3/5/8）

每个文件为 300 trial × 5 方法（median / minimum / maximum / rand_minmax / copy_paste）的经典 baseline 结果，指标含 ASR、R3_escape、R5_escape、ACC_near、ACC_near_norm 与音质 PESQ/STOI/SI-SDR。

| 相对路径 | 介绍 | 更改时间 |
|---|---|---|
| `data/baselines/baselines_audioseal_K2.csv` | audioseal 在 K=2 coalition 下的经典 baseline | 2026-08-17 17:48:25 |
| `data/baselines/baselines_audioseal_K3.csv` | audioseal 在 K=3 coalition 下的经典 baseline | 2026-08-17 17:48:25 |
| `data/baselines/baselines_audioseal_K5.csv` | audioseal 在 K=5 coalition 下的经典 baseline | 2026-08-17 17:48:25 |
| `data/baselines/baselines_audioseal_K8.csv` | audioseal 在 K=8 coalition 下的经典 baseline | 2026-08-17 17:48:25 |
| `data/baselines/baselines_timbrewm_K2.csv` | timbrewm 在 K=2 coalition 下的经典 baseline | 2026-08-17 17:48:25 |
| `data/baselines/baselines_timbrewm_K3.csv` | timbrewm 在 K=3 coalition 下的经典 baseline | 2026-08-17 17:48:25 |
| `data/baselines/baselines_timbrewm_K5.csv` | timbrewm 在 K=5 coalition 下的经典 baseline | 2026-08-17 17:48:25 |
| `data/baselines/baselines_timbrewm_K8.csv` | timbrewm 在 K=8 coalition 下的经典 baseline | 2026-08-17 17:48:25 |
| `data/baselines/baselines_voicemark_K2.csv` | voicemark 在 K=2 coalition 下的经典 baseline | 2026-08-17 17:48:25 |
| `data/baselines/baselines_voicemark_K3.csv` | voicemark 在 K=3 coalition 下的经典 baseline | 2026-08-17 17:48:25 |
| `data/baselines/baselines_voicemark_K5.csv` | voicemark 在 K=5 coalition 下的经典 baseline | 2026-08-17 17:48:25 |
| `data/baselines/baselines_voicemark_K8.csv` | voicemark 在 K=8 coalition 下的经典 baseline | 2026-08-17 17:48:26 |
| `data/baselines/baselines_wavmark_K2.csv` | wavmark 在 K=2 coalition 下的经典 baseline | 2026-08-17 17:48:25 |
| `data/baselines/baselines_wavmark_K3.csv` | wavmark 在 K=3 coalition 下的经典 baseline | 2026-08-17 17:48:25 |
| `data/baselines/baselines_wavmark_K5.csv` | wavmark 在 K=5 coalition 下的经典 baseline | 2026-08-17 17:48:25 |
| `data/baselines/baselines_wavmark_K8.csv` | wavmark 在 K=8 coalition 下的经典 baseline | 2026-08-17 17:48:25 |
| `data/baselines/baselines_wmcodec_K2.csv` | wmcodec 在 K=2 coalition 下的经典 baseline | 2026-08-17 17:48:26 |
| `data/baselines/baselines_wmcodec_K3.csv` | wmcodec 在 K=3 coalition 下的经典 baseline | 2026-08-17 17:48:26 |
| `data/baselines/baselines_wmcodec_K5.csv` | wmcodec 在 K=5 coalition 下的经典 baseline | 2026-08-17 17:48:26 |
| `data/baselines/baselines_wmcodec_K8.csv` | wmcodec 在 K=8 coalition 下的经典 baseline | 2026-08-17 17:48:26 |

## 篡改 TCT（payload-aware，5 模型 × K=2/3/5/8）

每个文件为 300 trial × 10 候选 target × 2 方法（mean / tct）的 payload-aware 篡改结果，指标为 target_top1。原方法名 `framing_cb` 已改为 `tct`（Targeted Convex Tampering），文件名沿用 `tamper_*.csv`（同一文件含 mean 与 tct 两种方法的行，与 `attack_*.csv` 含 mean/fwp 的组织方式一致）。

| 相对路径 | 介绍 | 更改时间 |
|---|---|---|
| `data/tamper/tamper_audioseal_K2.csv` | audioseal 在 K=2 coalition 下的 payload-aware 篡改（mean/tct） | 2026-08-17 17:48:25 |
| `data/tamper/tamper_audioseal_K3.csv` | audioseal 在 K=3 coalition 下的 payload-aware 篡改（mean/tct） | 2026-08-17 17:48:25 |
| `data/tamper/tamper_audioseal_K5.csv` | audioseal 在 K=5 coalition 下的 payload-aware 篡改（mean/tct） | 2026-08-17 17:48:25 |
| `data/tamper/tamper_audioseal_K8.csv` | audioseal 在 K=8 coalition 下的 payload-aware 篡改（mean/tct） | 2026-08-17 17:48:25 |
| `data/tamper/tamper_timbrewm_K2.csv` | timbrewm 在 K=2 coalition 下的 payload-aware 篡改（mean/tct） | 2026-08-17 17:48:25 |
| `data/tamper/tamper_timbrewm_K3.csv` | timbrewm 在 K=3 coalition 下的 payload-aware 篡改（mean/tct） | 2026-08-17 17:48:25 |
| `data/tamper/tamper_timbrewm_K5.csv` | timbrewm 在 K=5 coalition 下的 payload-aware 篡改（mean/tct） | 2026-08-17 17:48:25 |
| `data/tamper/tamper_timbrewm_K8.csv` | timbrewm 在 K=8 coalition 下的 payload-aware 篡改（mean/tct） | 2026-08-17 17:48:25 |
| `data/tamper/tamper_voicemark_K2.csv` | voicemark 在 K=2 coalition 下的 payload-aware 篡改（mean/tct） | 2026-08-17 17:48:25 |
| `data/tamper/tamper_voicemark_K3.csv` | voicemark 在 K=3 coalition 下的 payload-aware 篡改（mean/tct） | 2026-08-17 17:48:25 |
| `data/tamper/tamper_voicemark_K5.csv` | voicemark 在 K=5 coalition 下的 payload-aware 篡改（mean/tct） | 2026-08-17 17:48:25 |
| `data/tamper/tamper_voicemark_K8.csv` | voicemark 在 K=8 coalition 下的 payload-aware 篡改（mean/tct） | 2026-08-17 17:48:26 |
| `data/tamper/tamper_wavmark_K2.csv` | wavmark 在 K=2 coalition 下的 payload-aware 篡改（mean/tct） | 2026-08-17 17:48:25 |
| `data/tamper/tamper_wavmark_K3.csv` | wavmark 在 K=3 coalition 下的 payload-aware 篡改（mean/tct） | 2026-08-17 17:48:25 |
| `data/tamper/tamper_wavmark_K5.csv` | wavmark 在 K=5 coalition 下的 payload-aware 篡改（mean/tct） | 2026-08-17 17:48:25 |
| `data/tamper/tamper_wavmark_K8.csv` | wavmark 在 K=8 coalition 下的 payload-aware 篡改（mean/tct） | 2026-08-17 17:48:25 |
| `data/tamper/tamper_wmcodec_K2.csv` | wmcodec 在 K=2 coalition 下的 payload-aware 篡改（mean/tct） | 2026-08-17 17:48:26 |
| `data/tamper/tamper_wmcodec_K3.csv` | wmcodec 在 K=3 coalition 下的 payload-aware 篡改（mean/tct） | 2026-08-17 17:48:26 |
| `data/tamper/tamper_wmcodec_K5.csv` | wmcodec 在 K=5 coalition 下的 payload-aware 篡改（mean/tct） | 2026-08-17 17:48:26 |
| `data/tamper/tamper_wmcodec_K8.csv` | wmcodec 在 K=8 coalition 下的 payload-aware 篡改（mean/tct） | 2026-08-17 17:48:26 |

## 脉冲噪声对照

每个文件为按说话人聚合的 ASR 扫描，12 个 (eps, r0) 组合（eps ∈ {0, 0.001, 0.01, 0.05}，r0 ∈ {0.05, 0.1, 0.3}），38 说话人，n_trials=50（独立小规模探针，未随本轮 n=300 同步扩容）。

| 相对路径 | 介绍 | 更改时间 |
|---|---|---|
| `data/pulse_noise/pulse_noise_timbrewm_K5.csv` | timbrewm 在 K=5 下的脉冲噪声 ASR 扫描（mean 凸混合 + 两点分布噪声）| 2026-08-16 17:57:50 |
| `data/pulse_noise/pulse_noise_audioseal_K5.csv` | audioseal 在 K=5 下的脉冲噪声 ASR 扫描（mean 凸混合 + 两点分布噪声）| 2026-08-16 17:58:05 |

## 统计检验

`stats_all.txt` 为旧 n=150 数据上跑的检验结果，尚未随本轮 n=300 数据重新生成，使用时请注意版本不一致。

| 相对路径 | 介绍 | 更改时间 |
|---|---|---|
| `data/stats_all.txt` | 按说话人分层的 McNemar 检验 + paired bootstrap 置信区间，覆盖 5 模型 × 4K × ASR/ACC_near 两口径（**基于 n=150 旧数据，待更新**） | 2026-08-16 19:01:05 |

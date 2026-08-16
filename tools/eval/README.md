# 🎧 AudioEval — 音频质量评估工具

支持 **PESQ / SI-SNR / STOI / ViSQOL** 四个维度，批量评估一个文件夹下所有音频文件。

---

## 📁 项目结构

```
audioeval/
├── eval_metrics.py   # 各指标计算核心模块
├── run_eval.py       # 主程序入口
├── config.yaml       # 评估配置文件
├── requirements.txt  # Python 依赖
└── README.md
```

---

## 🚀 快速开始

### 第一步：安装基础依赖

```bash
pip install soundfile librosa pesq pystoi numpy scipy pandas tqdm PyYAML
```

### 第二步：安装 ViSQOL

**推荐方式（无需 Bazel）：**

```bash
pip install pyvisqol -i https://pypi.org/simple/
```

> `pyvisqol` 是轻量级封装，内置预编译二进制，**无需安装 Bazel**，直接 pip 即可使用。

**备选方式（官方库）：**

```bash
pip install visqol

cd visqol
# 使用 -f 参数确保下载失败时不保存文件
# curl -L -f -o bazelisk-linux-amd64 https://github.com/bazelbuild/bazelisk/releases/latest/download/bazelisk-linux-amd64
# chmod +x bazelisk-linux-amd64
# sudo cp bazelisk-linux-amd64 /usr/local/bin/bazel

# sudo yum update ca-certificates -y
# conda update -c conda-forge ca-certificates -y
# wget https://github.com/bazelbuild/bazel/releases/download/9.0.1/bazel-9.0.1-installer-linux-x86_64.sh
# ./bazel-9.0.1-installer-linux-x86_64.sh

# 3. 验证安装
bazel version
pip install numpy

# 4. 编译并安装
bazel build :visqol -c opt
pip install .
```

> 若两种方式都失败，可在 `config.yaml` 中将 `visqol` 从 metrics 列表中移除，跳过该指标。

### 第三步：修改配置文件

编辑 `config.yaml`，填写你的目录路径：

```yaml
ref_dir: /path/to/clean/audio      # 干净/参考语音目录
deg_dir: /path/to/processed/audio  # 待评估语音目录
```

### 第四步：运行评估

```bash
# 使用配置文件（推荐）
python run_eval.py --config config.yaml

# 或直接通过命令行参数
python run_eval.py \
    --ref_dir /path/to/ref \
    --deg_dir /path/to/deg \
    --metrics pesq sisnr stoi visqol \
    --output_csv results.csv \
    --output_json summary.json
```

---

## 📊 评估指标说明

| 指标 | 全称 | 范围 | 说明 |
|------|------|------|------|
| **PESQ** | Perceptual Evaluation of Speech Quality | [-0.5, 4.5] | 感知语音质量，越高越好 |
| **SI-SNR** | Scale-Invariant Signal-to-Noise Ratio | (-∞, +∞) dB | 尺度不变信噪比，越高越好 |
| **STOI** | Short-Time Objective Intelligibility | [0, 1] | 语音可懂度，越高越好 |
| **ViSQOL** | Virtual Speech Quality Objective Listener | [1, 5] | 感知音频质量，越高越好 |

### PESQ
- 需要参考音频（干净语音）
- 支持 **8000 Hz**（窄带，NB-PESQ）和 **16000 Hz**（宽带，WB-PESQ）
- 其他采样率会自动重采样到 16000 Hz

### SI-SNR
- 无需特定采样率，纯数值计算
- 对信号幅度缩放不敏感，适合评估语音分离/增强效果

### STOI
- 支持标准 STOI 和 Extended STOI（`stoi_extended: true`）
- ESTOI 在低 SNR 场景下更准确

### ViSQOL
- **speech 模式**（默认）：16 kHz，适合语音评估
- **audio 模式**：48 kHz，适合音乐/宽带音频评估
- 代码自动检测已安装的库（优先 `pyvisqol`，其次官方 `visqol`）

---

## 📦 ViSQOL 安装详解

### ✅ 方式一：pyvisqol（推荐，最简单）

```bash
pip install pyvisqol -i https://pypi.org/simple/
```

- 无需 Bazel，无需编译
- 内置预编译的 ViSQOL 二进制
- 支持 Linux / macOS / Windows

### 方式二：官方 visqol

```bash
pip install visqol
```

若提示 `bazel: command not found`，说明需要先安装 Bazel：

```bash
# Ubuntu/Debian
sudo apt install apt-transport-https curl gnupg -y
curl -fsSL https://bazel.build/bazel-release.pub.gpg | gpg --dearmor > bazel.gpg
sudo mv bazel.gpg /etc/apt/trusted.gpg.d/
echo "deb [arch=amd64] https://storage.googleapis.com/bazel-apt stable jdk1.8" | \
    sudo tee /etc/apt/sources.list.d/bazel.list
sudo apt update && sudo apt install bazel -y
```

### 方式三：跳过 ViSQOL

在 `config.yaml` 中注释掉 visqol：

```yaml
metrics:
  - pesq
  - sisnr
  - stoi
  # - visqol
```

---

## 📂 文件配对方式

### `match_by: filename`（默认，推荐）

按文件名（不含扩展名）匹配：

```
ref/001.wav  ↔  deg/001.wav
ref/002.flac ↔  deg/002.flac
```

### `match_by: order`

按排序顺序一一对应，适合文件名不同但顺序一致的情况：

```
ref/ 第1个文件  ↔  deg/ 第1个文件
ref/ 第2个文件  ↔  deg/ 第2个文件
```

---

## 📋 输出格式

### `results.csv` — 逐文件结果

```
ref_file,deg_file,pesq,sisnr,stoi,visqol
/ref/001.wav,/deg/001.wav,3.1245,12.3456,0.8921,3.8765
/ref/002.wav,/deg/002.wav,2.9876,10.1234,0.8654,3.7123
...
```

### `summary.json` — 汇总统计

```json
{
  "pesq_mean": 3.056,
  "pesq_std": 0.123,
  "pesq_min": 2.456,
  "pesq_max": 3.987,
  "sisnr_mean": 11.234,
  "sisnr_std": 1.456,
  ...
  "total_pairs": 100,
  "failed_pairs": 0,
  "failed_files": []
}
```

---

## ⚙️ 完整命令行参数

```
usage: run_eval.py [-h] [--config CONFIG]
                   [--ref_dir REF_DIR] [--deg_dir DEG_DIR]
                   [--output_csv OUTPUT_CSV] [--output_json OUTPUT_JSON]
                   [--metrics {pesq,sisnr,stoi,visqol} [...]]
                   [--match_by {filename,order}]
                   [--pesq_sr {8000,16000}]
                   [--stoi_extended]
                   [--visqol_audio_mode]

参数说明:
  --config            YAML 配置文件路径（优先级高于其他参数）
  --ref_dir           参考音频目录
  --deg_dir           待评估音频目录
  --output_csv        逐文件结果输出路径（默认: results.csv）
  --output_json       汇总统计输出路径（默认: summary.json）
  --metrics           要计算的指标，可多选（默认: 全部）
  --match_by          文件配对方式 filename/order（默认: filename）
  --pesq_sr           PESQ 采样率 8000/16000（默认: 16000）
  --stoi_extended     使用 Extended STOI
  --visqol_audio_mode ViSQOL 使用 audio 模式（默认: speech 模式）
```

---

## 🔧 支持的音频格式

`.wav` `.flac` `.mp3` `.ogg` `.m4a` `.aac`

---

## 📌 注意事项

1. **PESQ、STOI、SI-SNR 均需要参考音频**（干净语音）与待评估音频配对
2. **ViSQOL** 对音频时长有要求，过短的音频（< 1s）可能导致计算失败
3. 所有指标均会自动处理**多声道 → 单声道**转换（取均值）
4. 若某文件计算失败，该文件对应指标记为 `NaN`，不影响其他文件的评估
5. 汇总统计自动忽略 `NaN` 值
6. ViSQOL 代码会**自动检测**已安装的库（优先 `pyvisqol`，其次官方 `visqol`）

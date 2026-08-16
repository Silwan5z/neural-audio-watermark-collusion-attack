# 数据集目录

此目录存放实验用原始音频，**不随仓库分发**（已 .gitignore）。

## 需要的目录结构

```
dataset/
└── libritts16k/            # LibriTTS 16kHz 切片
    ├── 121_000000.wav      # 命名约定 {spk}_{file}.wav
    ├── 121_000001.wav
    ├── 237_000000.wav
    └── ...
```

- 下载：LibriTTS 可从 [OpenSLR](https://www.openslr.org/60) 获取，16kHz 版本见 `train-clean-*`。
- 命名约定见 `src/registry.py` 的 `clean_path_v19`：同一说话人下取时长最长的文件（避免 wavmark 嵌入的最小 chunk 长度断言失败）。

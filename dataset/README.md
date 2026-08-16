# 数据集目录

此目录存放实验用原始音频，**不随仓库分发**（已 .gitignore）。

## 生成 libritts16k

```bash
python tools/prepare_libritts16k.py \
    --libritts /path/to/LibriTTS/test-clean \
    --out dataset/libritts16k
```

- 输入：LibriTTS test-clean（[OpenSLR 60](https://www.openslr.org/60)），24kHz，目录结构 `{spk}/{book}/{spk}_{book}_{utt}_{seg}.wav`。
- 输出：`dataset/libritts16k/` 下 115 个 16kHz 单声道 wav，命名 `{spk}_{book}_{spk}_{book}_{utt}_{seg}.wav`。
- 脚本只处理清单列出的 115 个文件，输出与论文实验使用的音频完全一致。

加载约定见 `src/registry.py` 的 `clean_path_v19`：同一说话人下取时长最长的文件。

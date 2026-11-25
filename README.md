# D08 LEVIR-CD 第一次实验

本仓库归档第一次本机实验的 `strict_budget_v3` 源码、39 组原始训练控制台日志及逐轮记录。每组运行 100 轮，每轮 100 次参数更新。记录覆盖 5%、10%、20% 标注条件下的纯监督、半监督和边界消融，以及 100% 标注的纯监督参照。10% 标注下，历史半监督平均测试 F1 提升 1.68 个百分点，目标是 3 个百分点。

| 路径 | 内容 |
| --- | --- |
| `versions/strict_budget_v3` | 训练、数据读取、伪标签、评估及软件测试 |
| `scripts/research_queue_strict.py` | 历史训练队列 |
| `logs/console` | 39 份原始训练输出，每组 100 轮 |
| `logs/history` | 39 份逐轮 JSONL，包括按计划写入的验证指标 |
| `logs/configs` | 原始配置快照，保留历史机器路径 |
| `results` | 历史训练完成标记、测试汇总和独立验收记录 |
| `logs/SHA256SUMS.txt` | 归档文件内容校验值 |
| `docs/experiment-protocol.md` | 实验协议 |

## 在新目录运行

历史环境为 Python 3.9、PyTorch 2.8 和 Apple MPS。依赖版本见 `requirements.txt`。从 [LEVIR-CD 作者数据页](https://justchenhao.github.io/LEVIR/) 获取数据并按 `data/LEVIR-CD/{train,val,test}/{A,B,label}` 解压。原始上游网络文件从 [FC-Siam-diff 作者仓库](https://github.com/rcdaudt/fully_convolutional_change_detection) 获取，下载脚本校验与历史实验相同的 SHA-256。

```bash
python scripts/fetch_official_backbone.py
python versions/strict_budget_v3/main.py prepare --data-root data/LEVIR-CD --output data/manifests_strict
python scripts/run_matrix.py
```

最后一条命令预览 39 组计划。下面的命令运行完整训练，并用验证集选定的权重评估测试集和亮度对照。

```bash
python scripts/run_matrix.py --execute --max-parallel 2 --evaluate --illumination
```

重新运行的产物会写入 `runs_strict_v3`。仓库内的历史日志可独立阅读。数据、模型权重和虚拟环境由本地环境提供。

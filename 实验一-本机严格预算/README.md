# 实验一：本机严格预算实验

本目录保存第一次本机实验的 `strict_budget_v3` 冻结源码、39 组原始训练控制台日志、逐轮历史、运行配置和结果。每组计划运行 100 轮、每轮 100 次参数更新，包含 5%、10%、20% 标注下的纯监督、半监督和边界消融，以及 100% 标注的纯监督参照。10% 标注下半监督相对纯监督的历史平均测试 F1 提升为 1.6804 个百分点，未达到 3 个百分点目标。

## 运行环境

历史运行环境为 macOS、Apple MPS、Python 3.9 和 PyTorch 2.8.0。原机 Mac 的具体型号、CPU 和内存没有记录。依赖见本目录 `requirements-local-lock.txt` 和 `requirements.txt`。新运行建议用独立 Python 环境，避免改动系统 Python。

## 数据准备

从 [LEVIR-CD 作者数据页](https://justchenhao.github.io/LEVIR/) 获取数据。不要将数据上传到 GitHub。解压后目录应为：

```text
data/LEVIR-CD/
  train/{A,B,label}/
  val/{A,B,label}/
  test/{A,B,label}/
```

同一 split 下 A、B、label 文件名需配对。该实验运行所需的原始上游 FC-Siam-diff 网络文件不直接随仓库发布；`scripts/fetch_official_backbone.py` 会下载并检查 SHA-256。

## 准备和运行

在本目录作为当前目录执行：

```bash
python -m pip install -r requirements-local-lock.txt
python scripts/fetch_official_backbone.py
python versions/strict_budget_v3/main.py prepare --data-root data/LEVIR-CD --output data/manifests_strict
python scripts/run_matrix.py
```

末条命令只预览 39 组计划。确认数据和磁盘空间后，下面的命令开始完整队列，最多并行两个任务，并运行测试集和亮度对照评估：

```bash
python scripts/run_matrix.py --execute --max-parallel 2 --evaluate --illumination
```

新运行产物写入 `runs_strict_v3/`。仓库中的历史记录可独立审阅：`logs/console/` 为原始控制台输出，`logs/history/` 为逐轮 JSONL，`logs/configs/` 为配置快照，`results/` 为完成状态和结果汇总。原配置中保留了历史机器路径；在其他设备复跑时，以本说明中的相对路径为准并重新生成数据清单。

## 代码与数据构造文件清单

| 用途 | 仓库文件 | 作用 |
| --- | --- | --- |
| 数据下载辅助 | `scripts/download_data.py` | 获取数据的辅助脚本；也可从上面的作者页面自行获取。 |
| 数据构造入口 | `versions/strict_budget_v3/main.py` 的 `prepare` 命令 | 调用 `cd/prepare.py`，核对 train/val/test 中 A、B、label 的配对，生成数据清单。 |
| 数据读取 | `versions/strict_budget_v3/cd/data.py` | 按清单读取双时相图像与掩膜。 |
| 训练入口 | `versions/strict_budget_v3/main.py` 的 `train` 命令、`scripts/run_matrix.py` | 单组训练与完整矩阵调度；训练实现为 `cd/train.py`。 |
| 评估入口 | `versions/strict_budget_v3/main.py` 的 `evaluate` 命令 | 调用 `cd/evaluate.py` 与 `cd/metrics.py`；历史结果汇总在 `results/summary.csv` 和 `results/summary.json`。 |

## 核心实验：三个随机种子的完整结果

核心比较是 10% 标注下纯监督 A 组与半监督 B 组。下表 F1 为 `results/summary.csv` 中记录的测试 F1；差值单位为百分点。每个种子的两组训练完成记录分别为 `results/10pct_A_seed<种子>_training_complete.json` 与 `results/10pct_B_seed<种子>_training_complete.json`，其中 `epochs=100`。对应的原始训练输出、逐轮历史和配置分别位于 `logs/console/`、`logs/history/`、`logs/configs/`，文件名前缀均为 `10pct_A_seed<种子>` 或 `10pct_B_seed<种子>`。

| 种子 | A 组测试 F1 | B 组测试 F1 | B−A |
| ---: | ---: | ---: | ---: |
| 17 | 0.553084 | 0.572952 | +1.9868 |
| 42 | 0.491327 | 0.506486 | +1.5159 |
| 2026 | 0.605744 | 0.621129 | +1.5385 |

三种子平均提升为 **+1.6804 个百分点**，详见 `results/summary.json` 中 `percent=10, comparison=B-A` 的记录；没有达到预设的 +3 个百分点。第一次实验的历史归档包含代码、配置、日志和结果表，未包含原训练运行目录中的模型检查点与原始图像。`training_complete.json` 的 `test_evaluated=false` 是训练阶段的标记；测试指标记录在最终汇总表中，请勿把它理解为汇总表不存在测试结果。

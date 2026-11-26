# D08 第二次实验完整归档

本包包含三种子纯监督与晚介入半监督对照的代码、实际执行代码快照、配置、清单、训练日志、历史、结果和检查点。

种子 17 的 A 组为 项目/D08_PairedPilot15000/results/sl10，B 组为 项目/D08_LatePseudo15000/results/semi10。
种子 42 的 A、B 组在 项目/D08_ThreeSeeds/42/results。
种子 2026 的 A 组在 项目/D08_ThreeSeeds/2026/results/sl10，B 组在 项目/D08_ThreeSeeds/continuation-2026-B/results/semi10，含续跑日志。

正式测试材料位于 正式测试。protocol.json 固定六个最佳验证 F1 检查点、阈值、测试数据哈希和评估脚本哈希。summary.json 含逐组测试结果与三种子汇总。

LatePseudo15000/pilot.py 所需的 PairedPilot15000/results/summary.json、PairedPilot15000/results/sl10/history.json 和 PairedPilot15000/results/semi10/config.json 已包含。

本包不含训练、验证和封存测试图像，也不含 Python 虚拟环境。项目/assets.json 记录 train、val、test 三个数据包的数量、字节数和 SHA-256。正式测试重跑还需 sealed_test/test.zip，所需哈希载于 正式测试/protocol.json。恢复到另一台电脑时需按原项目根目录组织文件，并修改脚本的原机绝对路径。
MANIFEST.sha256.json 为每个归档成员的哈希。CHECKPOINTS.json 为六个检查点的可直接定位映射。

## 代码与数据构造文件清单

| 用途 | 仓库文件 | 作用 |
| --- | --- | --- |
| 数据构造与预检 | `项目/D08_PairedPilot15000/pilot.py prepare`、`项目/D08_LatePseudo15000/pilot.py prepare`、`项目/D08_ThreeSeeds/42/pilot.py prepare`、`项目/D08_ThreeSeeds/2026/pilot.py prepare` | 从 `项目/data/LEVIR-CD/` 核对配对图像与掩膜，重建各分支的数据清单；实现还调用 `项目/baseline/cd/prepare.py` 和 `项目/scripts/preflight.py`。旧清单含原机绝对路径，移机后须重新生成。 |
| 训练入口 | 上述各 `pilot.py run`、`项目/D08_ThreeSeeds/queue_runner.py run` | 启动种子 17、42、2026 的纯监督和晚介入半监督分支；核心训练实现为 `项目/scripts/worker.py` 和 `项目/baseline/cd/train.py`。 |
| 中断续跑 | `项目/D08_ThreeSeeds/continuation-2026-B/resume.py` | 种子 2026 的半监督分支从中断检查点续跑；`resume.log` 保存过程。 |
| 正式测试 | `正式测试/evaluate_frozen.py` | 对六个最佳验证检查点统一评估；`protocol.json` 固定数据与模型哈希，`summary.json` 保存结果。该脚本保留原机 `ROOT` 绝对路径，移机评估须在工作副本内修改，并在新的空评估目录运行。 |

完整环境与准备命令见仓库根目录 `README.md`。训练入口和正式测试脚本均已归档；数据包与 Python 环境需使用者自行准备。

## 核心实验：三个随机种子的完整结果

每个种子有纯监督 A 组与晚介入半监督 B 组，各运行 15,000 步。每组目录中均有 `result.json`（`completed=true`、`steps=15000`）、`history.json`、`config.json`、`best.pt` 和 `last.pt`。常规分支的训练输出位于对应子项目的 `results/sl10.log` 或 `results/semi10.log`；种子 2026 的 B 组还需结合原分支 `项目/D08_ThreeSeeds/2026/results/semi10.log` 与续跑目录的 `resume.log` 阅读。下表是 `正式测试/summary.json` 的测试 F1，不是验证集 F1。

| 种子 | A 组结果目录 | B 组结果目录 | A 组测试 F1 | B 组测试 F1 | B−A（百分点） |
| ---: | --- | --- | ---: | ---: | ---: |
| 17 | `项目/D08_PairedPilot15000/results/sl10/` | `项目/D08_LatePseudo15000/results/semi10/` | 0.694984 | 0.699002 | +0.4018 |
| 42 | `项目/D08_ThreeSeeds/42/results/sl10/` | `项目/D08_ThreeSeeds/42/results/semi10/` | 0.689517 | 0.682986 | −0.6531 |
| 2026 | `项目/D08_ThreeSeeds/2026/results/sl10/` | `项目/D08_ThreeSeeds/continuation-2026-B/results/semi10/` | 0.713254 | 0.702989 | −1.0265 |

种子 2026 的 B 组原运行在 12,044 步中断，因此最终 15,000 步结果取自 `continuation-2026-B`。三种子平均测试 F1 变化为 **−0.4259 个百分点**，未达到预设的 +3 个百分点。`正式测试/summary.json`、`正式测试/protocol.json`、`CHECKPOINTS.json` 与各组检查点共同提供测试结果、运行完成状态和模型文件的对应关系。

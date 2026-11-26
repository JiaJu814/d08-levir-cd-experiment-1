# 模式识别大作业：LEVIR-CD 建筑变化检测

仓库分为两个独立实验目录，保存代码、配置、日志、模型检查点和结果。两次实验都使用 LEVIR-CD 建筑变化检测数据和 FC-Siam-diff/Siamese 基线；第二次实验比较 10% 标注下的纯监督训练与晚介入半监督训练。

> **结果摘要**：第一次实验中，10% 标注的半监督组相对纯监督组平均测试 F1 提高 1.6804 个百分点，未达到预设的 3 个百分点目标。第二次实验三组配对种子的平均测试 F1 变化为 −0.4259 个百分点，也未达到目标。第二次测试协议注明：本次协议下测试集在训练/模型选择阶段未访问，但更早研究曾使用同一测试集；引用结果时须保留此限制。

## 目录

- [`实验一-本机严格预算/`](./实验一-本机严格预算/)：第一次本机实验的冻结代码、39 组训练日志、逐轮历史、配置和结果。
- [`实验二-Windows晚介入/`](./实验二-Windows晚介入/)：Windows 11 上第二次实验的六组最终配对训练、续跑记录、代码、检查点、正式测试脚本和汇总。归档不含 LEVIR-CD 图像和 Python 环境。

## 实验一：第一次本机实验

**设备与软件记录**：历史实验在 macOS 上使用 Apple MPS；Python 3.9、PyTorch 2.8.0。仓库没有记录这台 Mac 的具体 CPU、内存和型号，因此这里不补写型号。依赖锁定在 `实验一-本机严格预算/requirements-local-lock.txt`。

**数据与路径**：从 [LEVIR-CD 作者数据页](https://justchenhao.github.io/LEVIR/) 获取公开数据。将图像按下面的结构放在实验一目录中；数据不上传到本仓库。

```text
实验一-本机严格预算/
  data/LEVIR-CD/
    train/{A,B,label}/
    val/{A,B,label}/
    test/{A,B,label}/
```

**准备与运行**：在实验一目录作为当前目录执行。建议使用独立 Python 3.9 环境；安装 `requirements-local-lock.txt` 后，先获取作者网络源文件（脚本会核对 SHA-256），再准备数据清单。

```bash
cd 实验一-本机严格预算
python -m pip install -r requirements-local-lock.txt
python scripts/fetch_official_backbone.py
python versions/strict_budget_v3/main.py prepare --data-root data/LEVIR-CD --output data/manifests_strict
python scripts/run_matrix.py                 # 仅预览计划
python scripts/run_matrix.py --execute --max-parallel 2 --evaluate --illumination
```

完整矩阵会运行较长时间。新运行输出写入 `runs_strict_v3/`；仓库中的历史日志位于 `logs/`，结果汇总位于 `results/`。请先确认数据目录和磁盘空间，再运行完整队列。

## 实验二：Windows 11 晚介入实验

**实际设备**：Windows 11；Intel Core i7-14700KF、NVIDIA GeForce RTX 5060 Ti 16 GB、32 GB DDR5 内存。实际运行设备记录为 CUDA，预检记录显卡为 RTX 5060 Ti，并通过 CUDA 前向检查。归档中的训练入口明确要求 `torch==2.9.1+cu128`。Windows/CUDA 驱动版本和 Python 小版本没有被归档完整记录，不能据此声称某个精确版本曾在原机验证通过。

**数据与划分**：LEVIR-CD 训练集 445 对、验证集 64 对、测试集 128 对；训练使用 10% 标注（清单包含 44 对标注样本和未标注样本）。归档的 `assets.json` 记录原始压缩包 SHA-256、大小和配对检查信息，但没有附带数据。数据不上传本仓库。

将原始图像解压到实验二目录下：

```text
实验二-Windows晚介入/
  项目/
    data/LEVIR-CD/
      train/{A,B,label}/
      val/{A,B,label}/
      test/{A,B,label}/
    sealed_test/test.zip               # 重跑正式测试所需；外置数据，不入仓库
    .venv/Scripts/python.exe           # PowerShell 启动脚本所用环境
    baseline/  scripts/  D08_PairedPilot15000/
    D08_LatePseudo15000/  D08_ThreeSeeds/
  正式测试/
```

各 split 中 A、B 和 label 文件名必须配对。训练数据、验证数据及测试数据应来自官方 LEVIR-CD 对应划分。正式测试数据的 SHA-256 必须符合 `实验二-Windows晚介入/项目/assets.json` 及 `实验二-Windows晚介入/正式测试/protocol.json`；评估脚本内含原 Windows 机器绝对项目路径，换电脑后需在工作副本中更新其 `ROOT`，并核对本机 `sealed_test/test.zip` 哈希。不要用测试集挑选模型或调参。

**环境准备**：在 `实验二-Windows晚介入/项目/` 建立 `.venv`，安装归档里的 `requirements-win.txt`，并单独安装与 CUDA 12.8 匹配的 PyTorch 2.9.1。辅助依赖文件没有包含 PyTorch；不要用 CPU 版替代，因为预检会断言 CUDA 和精确的 PyTorch 版本。

```powershell
cd .\实验二-Windows晚介入\项目
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r .\requirements-win.txt
# 按 PyTorch 官方安装选择器安装 PyTorch 2.9.1 + CUDA 12.8 对应构建
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CUDA unavailable')"
```

PyTorch 官方安装选择器：[https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)。继续前应确认输出版本为 `2.9.1+cu128`、CUDA 可用且识别到 NVIDIA 显卡。`项目/` 下各 `results/manifests/*.json` 保存了原机绝对数据路径；移机后应先运行各子项目的 `prepare` 重新生成本机清单和预检结果，不要直接沿用旧清单。

**运行入口与实验参数**：原始训练脚本位于 `D08_PairedPilot15000/`、`D08_LatePseudo15000/` 和 `D08_ThreeSeeds/`。PowerShell 启动脚本会使用 `实验二-Windows晚介入/项目/.venv`。三个配对种子为 17、42、2026；每组比较纯监督与晚介入半监督，目标步数 15,000。主要配置包括裁剪尺寸 256、batch size 4、验证间隔 500 步、EMA 系数 0.99、伪标签高/低阈值 0.95/0.01、晚介入 warm-up 9,000 步及 2,000 步 ramp。种子 2026 的半监督组曾在 12,044 步中断，归档同时包含断点和续跑到 15,000 步的记录。

**重新训练提示**：下列命令针对另建的干净工作副本。归档目录已经包含完整 `results/`，请保留它作为历史证据，不要直接在归档副本里覆盖结果。先将数据放到 `项目/data/LEVIR-CD/`，再在 `项目/` 根目录为各分支准备数据清单。例如：

```powershell
cd .\实验二-Windows晚介入\项目
.\.venv\Scripts\python.exe .\D08_PairedPilot15000\pilot.py prepare
.\.venv\Scripts\python.exe .\D08_LatePseudo15000\pilot.py prepare
.\.venv\Scripts\python.exe .\D08_ThreeSeeds\42\pilot.py prepare
.\.venv\Scripts\python.exe .\D08_ThreeSeeds\2026\pilot.py prepare
```

准备通过后，可在各项目子目录调用 `Run-D08Pilot.ps1 -Mode Start` 启动对应运行；42 和 2026 的队列入口为 `D08_ThreeSeeds/queue_runner.py`。正式重跑前请在副本中备份并移走对应旧 `results/`，不要覆盖本仓库中的原始记录。这些脚本保留原实验的预算、依赖和状态检查。建议先查看相应 `pilot.py status`、`scripts/preflight.py` 和状态记录，再按分支逐项启动；不要在首次运行时把所有种子并行启动。2026 种子的半监督续跑需按 `D08_ThreeSeeds/continuation-2026-B/resume.py` 使用中断检查点；从零复跑时请先阅读该脚本的输入/输出位置，并分别保存中断与续跑日志。

训练记录和检查点可直接查看，不需要安装环境。`实验二-Windows晚介入/正式测试/summary.json` 是已完成测试的汇总；只有在准备好相同测试集、核验哈希并按本机路径修改评估脚本 `ROOT` 后，才在一份新的干净 `正式测试/` 工作目录运行 `evaluate_frozen.py` 重算（脚本会拒绝覆盖已有 protocol/summary）。归档包含原始测试汇总和检查点，但没有逐像素预测图；数据包未随仓库发布。

## 公开与复核说明

- 两个实验目录均保留各自的配置、训练历史及结果；不要把两个实验的模型权重或指标混作同一批结果。
- 实验二完整归档的 SHA-256 清单在 `实验二-Windows晚介入/ARCHIVE_SHA256SUMS.txt`；原始归档内的 `MANIFEST.sha256.json` 也保留。
- 本仓库不包含 LEVIR-CD 图像、虚拟环境、CUDA 驱动或原始机器上的数据缓存。使用者需自行从数据来源取得数据并确认许可与哈希。
- 检查点和日志用于审阅既有实验。复现运行会产生新的结果，不应覆盖原归档记录。

# D08 第二次实验完整归档

本包包含三种子纯监督与晚介入半监督对照的代码、实际执行代码快照、配置、清单、训练日志、历史、结果和检查点。

种子 17 的 A 组为 项目/D08_PairedPilot15000/results/sl10，B 组为 项目/D08_LatePseudo15000/results/semi10。
种子 42 的 A、B 组在 项目/D08_ThreeSeeds/42/results。
种子 2026 的 A 组在 项目/D08_ThreeSeeds/2026/results/sl10，B 组在 项目/D08_ThreeSeeds/continuation-2026-B/results/semi10，含续跑日志。

正式测试材料位于 正式测试。protocol.json 固定六个最佳验证 F1 检查点、阈值、测试数据哈希和评估脚本哈希。summary.json 含逐组测试结果与三种子汇总。

LatePseudo15000/pilot.py 所需的 PairedPilot15000/results/summary.json、PairedPilot15000/results/sl10/history.json 和 PairedPilot15000/results/semi10/config.json 已包含。

本包不含训练、验证和封存测试图像，也不含 Python 虚拟环境。项目/assets.json 记录 train、val、test 三个数据包的数量、字节数和 SHA-256。正式测试重跑还需 sealed_test/test.zip，所需哈希载于 正式测试/protocol.json。恢复到另一台电脑时需按原项目根目录组织文件，并修改脚本的原机绝对路径。
MANIFEST.sha256.json 为每个归档成员的哈希。CHECKPOINTS.json 为六个检查点的可直接定位映射。

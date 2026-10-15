# examples/code_r1

Code-R1 的脚本、数据构造工具与奖励实现。**完整使用说明见仓库根目录的 [`README.md`](../../README.md)**（环境、数据、训练参数表、沙箱、导出、数据格式）。

## 文件说明

| 文件 | 用途与用法 |
|---|---|
| `run_code_r1_grpo_fsdp.sh` | GRPO 训练启动脚本（主结果）。`MODEL_PATH=<模型> DATA_DIR=<数据目录> bash examples/code_r1/run_code_r1_grpo_fsdp.sh`，可调环境变量见根 README 第 3 节。 |
| `code_reward_manager.py` | 自定义奖励管理器 `CodeR1RewardManager`：给单个 reward worker 内的沙箱并发加信号量。由 `reward.reward_manager.module.path` 指向本文件使用。 |
| `data_preprocess/download_code_r1_12k.py` | 下载官方 12K 数据集，并用沙箱复跑参考解做校验。`python examples/code_r1/data_preprocess/download_code_r1_12k.py --local_dir ./data/code-r1-12k --verify 64` |
| `data_preprocess/coder1.py` | 从 LeetCodeDataset + TACO 构造训练数据（参考解过不了测试的样本会被丢弃）。`python examples/code_r1/data_preprocess/coder1.py --root_dir ./data --leetcode_dir ./LeetCodeDataset` |
| `scripts/test_sandbox.py` | 沙箱功能测试与并发压测。`CODER1_EXEC=firejail python examples/code_r1/scripts/test_sandbox.py --stress 512 --concurrency 64` |
| `tests/test_coder1_reward.py` | 奖励单测（含官方真实样本端到端）。`CODER1_EXEC=local pytest -q examples/code_r1/tests/` |

## 相关代码

奖励实现在 `verl/utils/reward_score/coder1/`（`__init__.py` 判分主体、`firejail_exec.py` / `local_exec.py` 两种沙箱后端、`utils.py`），由 `verl/utils/reward_score/__init__.py` 按 `data_source == "code"` 分派。

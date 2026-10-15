# Code-R1

在 [verl](https://github.com/verl-project/verl) 上复现 **Code-R1**（R1-Zero for code）：奖励只用"单元测试过没过"这种二元可验证信号，代码在沙箱里执行，用 GRPO 训练代码模型。

本文档只讲怎么用，所有脚本在 `examples/code_r1/`。

---

## 1. 环境准备

```bash
pip install -e .
pip install vllm flash-attn wandb gpustat --no-build-isolation
```

训练机（Linux）建议装沙箱：

```bash
sudo add-apt-repository ppa:deki/firejail && sudo apt-get update && sudo apt-get install firejail firejail-profiles
```

要求：单机 8×GPU（`run_code_r1_grpo_fsdp.sh` 的默认值，可用 `GPUS_PER_NODE` 改）+ vLLM。
没有 Linux/沙箱也能验证数据与奖励链路，用 `CODER1_EXEC=local` 即可（见第 5 节）。

---

## 2. 数据准备

**方式 A（推荐）：直接下载官方 12K**。它已经是 verl 的 RL 格式，下载时用沙箱复跑一遍参考解筛掉坏样本：

```bash
python examples/code_r1/data_preprocess/download_code_r1_12k.py \
    --local_dir ./data/code-r1-12k --verify 64 --min_pass_rate 0.9
```

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--repo_id` | `ganler/code-r1-12k` | HuggingFace 数据集 |
| `--local_dir` | `./data/code-r1-12k` | 输出目录，写 `train.parquet` / `test.parquet` |
| `--verify` | `32` | 用沙箱复跑多少条参考解，`0` = 跳过 |
| `--min_pass_rate` | `0.0` | 复验通过率低于此值直接报错退出 |

**方式 B：从零构造**（LeetCode 2k + 沙箱验证过的 TACO 10k）：

```bash
git clone https://github.com/newfacade/LeetCodeDataset.git

python examples/code_r1/data_preprocess/coder1.py \
    --root_dir ./data --leetcode_dir ./LeetCodeDataset
```

输出 `./data/code-r1-<N>k-leetcode-taco/{train,test}.parquet`。常用参数：`--datasets leetcode taco`（可只选子集）、`--max_n_tests 8`、`--num_proc 8`。
构造时参考解跑不过测试的样本会被丢掉，所以最终条数会明显少于原始数据（26K TACO → 约 10K）。

---

## 3. 训练

```bash
MODEL_PATH=Qwen/Qwen2.5-7B-Instruct-1M \
DATA_DIR=./data/code-r1-12k \
    bash examples/code_r1/run_code_r1_grpo_fsdp.sh
```

可调的环境变量（都有默认值）：

| 变量 | 默认值 | 说明 |
|---|---|---|
| `GPUS_PER_NODE` | `8` | 单机 GPU 数 |
| `MODEL_PATH` | `Qwen/Qwen2.5-7B-Instruct-1M` | actor 模型路径 |
| `DATA_DIR` | `$REPO_ROOT/data/code-r1-12k` | 数据目录（读 `train.parquet` / `test.parquet`） |
| `MAX_EPOCHS` | `8` | 训练轮数 |
| `ROLLOUT_N` | `16` | 每个 prompt 采样条数 |
| `TRAIN_BATCH_SIZE` | `16` | 每步多少个 prompt |
| `MICRO_BATCH_PER_GPU` | `8` | actor 的 per-GPU micro batch |
| `GRAD_ACC_STEPS` | `4` | 梯度累积步数 |
| `MAX_PROMPT_LENGTH` | `2048` | prompt 最大长度 |
| `MAX_RESPONSE_LENGTH` | `4096` | 生成最大长度 |
| `LR` | `5e-7` | 学习率 |
| `REWARD_NUM_WORKERS` | `8` | 奖励计算并行度（reward loop worker 数） |
| `CODER1_EXEC` | `firejail` | 沙箱后端，`firejail` / `local` |
| `CODER1_TIMEOUT` | `20` | 单次沙箱执行超时（秒） |
| `CODE_REWARD_CONCURRENCY` | `NUM_CPUS / REWARD_NUM_WORKERS` | 单个 reward worker 内并发的沙箱数上限 |
| `CODE_REWARD_RETRIES` | `1` | 执行失败重试次数（无退避） |
| `NUM_CPUS` | `nproc` | 用于推算 `CODE_REWARD_CONCURRENCY` |
| `PROJECT_NAME` / `EXPERIMENT_NAME` | `code-r1` / `code-r1-12k-grpo` | wandb 与 checkpoint 命名 |

**一条约束**：`TRAIN_BATCH_SIZE × ROLLOUT_N` 必须能被 `GPUS_PER_NODE × MICRO_BATCH_PER_GPU × GRAD_ACC_STEPS` 整除，否则脚本直接报错退出。

**PPO 版本**：把 `algorithm.adv_estimator=grpo` 改成 `gae`，并补上 critic 那一组参数（新版在 `critic.fsdp.*`，不是 `critic.model.fsdp_config.*`）：

```bash
    algorithm.adv_estimator=gae \
    critic.optim.lr=1e-5 \
    critic.model.path=$MODEL_PATH \
    critic.model.use_remove_padding=True \
    critic.model.enable_gradient_checkpointing=True \
    critic.ppo_micro_batch_size_per_gpu=16 \
    critic.fsdp.param_offload=False \
    critic.fsdp.optimizer_offload=True \
```

---

## 4. 沙箱后端

| `CODER1_EXEC` | 隔离 | 适用场景 |
|---|---|---|
| `firejail`（Linux 默认） | seccomp 禁 socket + rlimit 限进程/文件/内存/时长 | 训练机首选 |
| `local` | 无隔离 | 本机调试、数据校验、非 Linux 开发机 |

沙箱相关环境变量只有三个：`CODER1_EXEC`（后端）、`CODER1_TIMEOUT`（默认 30s，训练脚本里设成 20）、`CODER1_LOG`（默认 1，是否打印每条样本的判分明细）。

`CODE_REWARD_CONCURRENCY × REWARD_NUM_WORKERS` 就是整机同时开的沙箱数，建议按物理核数来设；调得比核数大只会制造假超时。

---

## 5. 训练前自检

```bash
# 沙箱功能测试 + 并发压测（看有没有假失败）
CODER1_EXEC=firejail python examples/code_r1/scripts/test_sandbox.py --stress 512 --concurrency 64

# 奖励单测（含真实样本端到端：参考解 1.1、故意改错 0.1）
CODER1_EXEC=local pytest -q examples/code_r1/tests/
```

数据侧的自检就是第 2 节方式 A 里的 `--verify` / `--min_pass_rate`：通过率明显低于 ~90% 就说明奖励链路或沙箱有问题，先别开训。

---

## 6. 导出模型

```bash
python -m verl.model_merger merge --backend fsdp \
    --local_dir checkpoints/code-r1/code-r1-12k-grpo/global_step_800 \
    --target_dir out
```

`--local_dir` 指向含 `fsdp_config.json` 的那一层（本版本是 `<ckpt>/global_step_N`）。

---

## 7. 奖励怎么接进去

训练脚本用的是**自定义 reward manager**；仓库里同时保留了**改核心**的方式，二选一即可。

**自定义 reward manager**（`run_code_r1_grpo_fsdp.sh` 用的就是这个）：

```bash
    reward.reward_manager.source=importlib \
    reward.reward_manager.name=CodeR1RewardManager \
    reward.reward_manager.module.path=$REPO_ROOT/examples/code_r1/code_reward_manager.py
```

它给单个 reward worker 内的沙箱并发加了信号量，并发上限由 `CODE_REWARD_CONCURRENCY` 控制。

**改核心**：`verl/utils/reward_score/__init__.py` 已经把 `data_source == "code"` 路由到 `verl/utils/reward_score/coder1/`，数据集字段叫 `code` 时开箱可用，不需要额外配置。

---

## 8. 数据格式

用自己的数据训练时，每条样本长这样：

```
data_source  = "code"                                            # 必需，奖励按它分派
prompt       = [{"role": "system", ...}, {"role": "user", ...}]
reward_model = {"style": "rule", "ground_truth": "<JSON 测试>"}   # 奖励读 ground_truth
extra_info   = {"split","index","prompt","reference","dataset"}   # 可选
```

`ground_truth` 支持三种形态：

- `{"functional": "<追加在解答后的测试>"}`：LeetCode 风格；
- `{"pytest": "<test_solution.py 的内容>"}`；
- `{"inputs": [...], "outputs": [...]}`：stdin/stdout 风格。

模型输出必须在最后一个 `<answer>` 里给出 ```` ``` ```` 围栏代码块，否则拿不到分。

---

## 9. 目录结构

```
examples/code_r1/
├── run_code_r1_grpo_fsdp.sh        GRPO 训练启动脚本（主结果）
├── code_reward_manager.py          自定义奖励管理器（沙箱并发限流）
├── data_preprocess/
│   ├── download_code_r1_12k.py     下载官方 12K + 沙箱复验参考解
│   └── coder1.py                   从 LeetCodeDataset / TACO 构造数据
├── scripts/test_sandbox.py         沙箱功能测试 + 并发压测
└── tests/                          奖励单测（本机可跑）
verl/utils/reward_score/coder1/     奖励实现（沙箱执行 + 判分）
```

---

## 10. 许可与来源

- 本仓库基于 [verl](https://github.com/verl-project/verl)（Apache-2.0）修改，`LICENSE` 保留原始版权声明；方法与数据来自 [ganler/code-r1](https://github.com/ganler/code-r1)。
- `data/`、`models/`、`*.parquet`、`*.safetensors` 等已在 `.gitignore` 中忽略，数据集与模型权重需自行下载。

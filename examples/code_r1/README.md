# Code-R1 复现笔记：如何改动 verl 来复现 Code-R1

把官方 [Code-R1](https://github.com/ganler/code-r1) 相对 verl 的改动，映射到当前 verl（`0.10.0.dev`）。
先说结论性的一件事：

> **官方 Code-R1 是 verl 的一个 fork，不是 verl 主干。** 它的核心文件（`examples/data_preprocess/coder1.py`、
> `verl/utils/reward_score/coder1/`、`main_grpo.sh`、`LeetCodeDataset`）在 verl 主仓里一个都没有。
> 本目录是把上游那套取回来、适配到新版 verl 的结果。

官方 diff 的取法（fork 点 `65cceb3` → HEAD `443f8da`，共 21 个文件 / +1525 −152）：

```bash
git clone https://github.com/ganler/code-r1.git && cd code-r1
git remote add upstream https://github.com/volcengine/verl.git && git fetch upstream main
git merge-base HEAD upstream/main      # -> 65cceb3
git diff --stat 65cceb3 HEAD
```

---

## 一、官方 Code-R1 改了什么

| 文件 | 改动 | 为什么 |
|---|---|---|
| `verl/utils/reward_score/coder1/*` | 新增：奖励主体 + 4 种沙箱后端 | 抽 `<answer>` 里的代码 → 沙箱跑单元测试 → 过给 `+1.1`、跑挂 `+0.1`、格式错 `−1.1` |
| `verl/utils/reward_score/__init__.py` | **+3 行** | 把 `data_source == "code"` 路由到上面的奖励 |
| `verl/workers/reward_manager/prime.py` | 进程池→线程池、并发 `cpu_count()`、超时 300→600s | 代码奖励是 I/O 密集（等沙箱），并发/超时不当会制造错误奖励 |
| `examples/data_preprocess/coder1.py` | 新增 381 行 | 造数据，**并用沙箱重跑参考解筛掉坏样本**（26K TACO 只剩 10K） |
| `examples/data_preprocess/kodcode.py` | 新增 | 另一个数据源（本目录未用） |
| `main_grpo.sh` / `main_ppo.sh` | 新增 | GRPO 是主结果，PPO 是对照 |
| `scripts/model_merger.py` | +83 行 | 合并 ckpt 时把 SYSTEM_PROMPT 写进 chat template，保证推理与训练 prompt 一致 |
| `scripts/test_sandbox.py`、`stress_exec.py` | 新增 | 训练前验证/压测沙箱 |

归纳成三句话，这才是 Code-R1 值得学的地方：

1. **奖励要二元、可验证**：不做 LLM 判分，只问"单元测试过没过"。
2. **数据要先用沙箱验一遍**：参考解跑不过测试的样本直接丢。
3. **执行要可靠**：高并发下沙箱因为超时/OS 限制误判，就是错误奖励（官方统称 reward false positive）。

---

## 二、本目录改了什么

### 2.1 verl 核心改动（3 处，路径与官方一致，便于对照 diff）

| 文件 | 改动 |
|---|---|
| `verl/utils/reward_score/coder1/` | **新增 4 个文件**：`__init__.py`（奖励主体）、`utils.py`、`firejail_exec.py`（训练机）、`local_exec.py`（本机，非沙箱） |
| `verl/utils/reward_score/__init__.py` | 加 `elif data_source in ["code"]:` → 调 `coder1.compute_score(...)`（与官方那 3 行一致） |
| `verl/workers/reward_manager/prime.py` | 官方那处线程池改动原样保留（它在新版已不在主路径，见 3.1） |

相对官方做了两处**减法**，理由是"用不上"：上游的 `docker` / `ces` / `kira` 三个后端要么需要另起服务、
要么官方自己实测假超时率高（`ces` 约 10%），所以只留 firejail 与 local；`codecontests` / `kodcode`
两个数据源也不在官方 12K 里，一并删掉。

### 2.2 recipe 文件（`examples/code_r1/`）

```
examples/code_r1/
├── README.md                     本文
├── run_code_r1_grpo.sh           官方 main_grpo.sh 的新版写法（GRPO，主结果）
├── code_reward_manager.py        code-r1 并发控制思路在现代 reward loop 里的落点
├── data_preprocess/
│   ├── download_code_r1_12k.py   下载官方 12K + 用沙箱复验参考解（最省事的起步）
│   └── coder1.py                 官方数据构造脚本（LeetCode 2k + 验证过的 TACO 10k）
├── scripts/test_sandbox.py       沙箱功能测试 + 并发压测
└── tests/                        6 个奖励单测（含官方真实样本端到端），本机可跑
```

> 路径映射：官方 `examples/data_preprocess/coder1.py` → 这里 `examples/code_r1/data_preprocess/coder1.py`；
> 官方根目录的 `main_*.sh` → 这里 `examples/code_r1/run_*.sh`。**奖励代码路径保持与官方一致**。
> 另外 verl 主仓的 `recipe/` 是未初始化的 git 子模块（`verl-project/verl-recipe`），写进去 git 看不见，所以放在 `examples/`。

---

## 三、版本迁移对照表（最容易翻车的地方）

| code-r1（2025-03） | verl `0.10.0.dev` | 说明 |
|---|---|---|
| `reward_model.reward_manager=prime` | `reward.reward_manager.name=naive` | 奖励管理器搬到 `reward.reward_manager.*`，注册表也换了（见 3.1） |
| `reward_model.reward_loop_source=...` | `reward.reward_manager.source=importlib` | 换自定义奖励管理器的入口 |
| `sandbox_fusion.url` | `reward.sandbox_fusion.url` | 云端沙箱 |
| （无） | `reward.num_workers=8` | **新增**：奖励计算的并行度（reward loop worker 数） |
| `...rollout.log_prob_micro_batch_size` | `...rollout.log_prob_micro_batch_size_per_gpu` | 改名 |
| `critic.model.fsdp_config.param_offload` | `critic.fsdp.param_offload` | critic 的 FSDP 配置挪到 `critic.fsdp.*` |
| `actor_rollout_ref.rollout.name` 有默认值 | 必填（`???`） | 必须显式写 `vllm` / `sglang` |

**改脚本时怎么确认 key 没写错**（不用占 GPU）：

```bash
python scripts/print_cfg.py --cfg job            # 打印合并后的完整配置
# 升级 verl 后 schema 变了：bash scripts/generate_trainer_config.sh
```

**顺带记住的两件事**：① 官方 `data_source == "code"` 的分派逻辑在新版一行都没挪，
`default_compute_score()` 仍是按 `data_source` 分派的大 if/elif；② 数据格式也没变，
官方 12K 数据集开箱即用：

```
data_source  = "code"                                            # 必需，奖励分派用
prompt       = [{"role": "system", ...}, {"role": "user", ...}]
reward_model = {"style": "rule", "ground_truth": "<JSON 测试>"}   # 奖励读 ground_truth
extra_info   = {"split","index","prompt","reference","dataset"}   # 可选
```

`ground_truth` 三种形态：`{"functional": "<追加在解答后的测试>"}`（LeetCode 风格）、
`{"pytest": "<test_solution.py 内容>"}`、`{"inputs": [...], "outputs": [...]}`（stdin/stdout 风格）。

### 3.1 为什么 `prime.py` 改了却"不算数"

新版 `PPOTrainer` 的奖励计算整体搬进了 **reward loop**：`RewardLoopManager` 起 `reward.num_workers`
个 Ray actor（`verl/experimental/reward_loop/`），奖励管理器只从
`verl/experimental/reward_loop/reward_manager` 这个注册表里取（`naive/dapo/gdpo/limited/remote`）。
`verl/workers/reward_manager/`（含 `prime`）已是历史遗留，`reward.reward_manager.name=prime` 会直接报
`Unknown reward manager`。

所以：① `prime.py` 里那处线程池改动我**保留**了，它就是你要看的 diff，也是理解"代码奖励为什么不能用进程池"
的最好材料（文件里也加了注释说明它已不在主路径）；② 它的**现代落点**在
`code_reward_manager.py`——新版"并行度"已有 `reward.num_workers` 这个旋钮，但**没人限制单个 worker 一次
起多少沙箱**，而那正是容易产生假超时的地方。

---

## 四、把奖励接进 verl 的两种方式

**A. 改核心**（官方做法）：在 `verl/utils/reward_score/__init__.py` 里按 `data_source` 注册。
适合数据集字段已经叫 `code`、想"像内置奖励一样"用它。本目录已经改好。

**C. 自定义 reward manager**（换掉"怎么算"）：

```bash
reward.reward_manager.source=importlib
reward.reward_manager.name=CodeR1RewardManager
reward.reward_manager.module.path=$REPO_ROOT/examples/code_r1/code_reward_manager.py
```

新版奖励管理器继承 `verl.experimental.reward_loop.reward_manager.base.RewardManagerBase`，实现
`async def run_single(self, data) -> {"reward_score": float, "reward_extra_info": {...}}`。
`reward_extra_info` 里的键会进批量数据，可以直接在 wandb 上看——这是给"奖励质量"做监控的地方。

---

## 五、沙箱与验证

### 5.1 两个后端

| `CODER1_EXEC` | 隔离 | 用途 |
|---|---|---|
| `firejail`（Linux 默认） | seccomp 禁 socket + rlimit 限进程/文件/内存/时长 | 训练机首选（官方结论） |
| `local` | ❌ 无隔离 | 本机调试、数据校验、非 Linux 开发机 |

安装 firejail：`sudo add-apt-repository ppa:deki/firejail && sudo apt-get update && sudo apt-get install firejail firejail-profiles`

环境变量只有三个：`CODER1_EXEC`（后端）、`CODER1_TIMEOUT`（默认 30s）、`CODER1_LOG`（默认 1，是否打印每条样本的判分明细）。

### 5.2 训练前必做的验证

```bash
# a) 沙箱体检；压测并发，看有没有假失败
CODER1_EXEC=firejail python examples/code_r1/scripts/test_sandbox.py --stress 512 --concurrency 64

# b) 数据和沙箱一起验：下载官方 12K 并用沙箱复跑参考解
python examples/code_r1/data_preprocess/download_code_r1_12k.py \
    --local_dir ./data/code-r1-12k --verify 64 --min_pass_rate 0.9
```

(b) 通过率明显低于 ~90% 就说明奖励链路或沙箱有问题，先别训。

---

## 六、跑起来

```bash
# 0) 依赖
pip install -e . && pip install vllm flash-attn wandb gpustat --no-build-isolation

# 1) 数据（二选一）
python examples/code_r1/data_preprocess/download_code_r1_12k.py --local_dir ./data/code-r1-12k --verify 64
# 或者从零造：git clone https://github.com/newfacade/LeetCodeDataset.git 后
#   python examples/code_r1/data_preprocess/coder1.py --root_dir ./data --leetcode_dir ./LeetCodeDataset

# 2) 训练（8 GPU）
MODEL_PATH=./models/Qwen2.5-7B-Instruct-1M DATA_DIR=./data/code-r1-12k \
    bash examples/code_r1/run_code_r1_grpo.sh
```

**想要 PPO 版本**（官方 `main_ppo.sh`）只需在 GRPO 那行命令上改三处：
`algorithm.adv_estimator=gae`（去掉 `use_kl_in_reward` 之外保持不变即可），加上 critic 那一组
（注意新版是 `critic.fsdp.*`，不是 `critic.model.fsdp_config.*`）：

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

**导出**：官方在 `scripts/model_merger.py` 里额外做了一件事——把 SYSTEM_PROMPT 写进
`tokenizer_config.json` 的 chat template，这样推理端只发 user 消息也能得到训练时的同款 prompt。
新版 merger 已搬到 `verl/model_merger`，所以那步要自己加：

```bash
python -m verl.model_merger merge --backend fsdp \
    --local_dir checkpoints/code-r1/code-r1-12k-grpo/global_step_800 --target_dir out
# --local_dir 指向含 fsdp_config.json 的那层（本版本是 <ckpt>/global_step_N）
```

---

## 七、常见坑

1. **`ground_truth` 为空 → 全是 0 分**：`code` 这个 data_source 要求它是 JSON 测试。
   现在会打印醒目告警后记 0，不再静默。
2. **代码块格式必须匹配**：官方奖励从最后一个 `<answer>` 里找 ```` ``` ```` 围栏代码块，
   `<answer>` 里没有代码块也拿不到分。prompt 与奖励解析必须对齐。
3. **`data_source` 拼错或大小写不对** → `NotImplementedError`。
4. **相对路径**：自定义奖励管理器的 `path` 在 Ray worker 里解析，用 `$REPO_ROOT` 拼绝对路径。
5. **并发与超时**：CPU 不够时调大 `reward.num_workers` 只会制造假超时，
   建议 `reward.num_workers × CODE_REWARD_CONCURRENCY ≈ 物理核数`。
6. **测试集泄漏**：官方 `coder1.py` 会按 `question_id` 去重并排除测试集题目，自己造数据别漏这步。
7. **`trainer.val_before_train=True`** 会先跑一遍评测，每条都要过沙箱，很慢，调参时先关掉。

---

## 八、本机验证记录

环境：Windows + Python 3.13，无 GPU、无 Ray ⇒ 只验证"不需要训练"的部分。

| 验证项 | 结果 |
|---|---|
| `CODER1_EXEC=local pytest -q examples/code_r1/tests/` | ✅ 6 passed（含官方 12K 真实样本端到端：参考解 `1.1`、故意改错 `0.1`） |
| `python examples/code_r1/scripts/test_sandbox.py` | ✅ 功能测试全过（含 pytest 收集、stdin、报错分支、LeetCode 追加测试） |
| `... --stress 16 --concurrency 8` | ✅ 16/16，约 8 exec/s |
| 启动脚本的 config key 逐条比对 schema | ✅ 全部有效（过程中抓出 `critic.model.fsdp_config.*` 与漏写 `+` 两处真错误） |
| `python -m compileall` + 行宽 ≤120 检查 | ✅ 通过 |
| 需要 GPU 才能验证的 | ⏳ 真实训练、firejail 压测、12K 下载与参考解复验（见 5.2） |

---

## 九、参考

* Code-R1 <https://github.com/ganler/code-r1> ｜ 数据 <https://huggingface.co/datasets/ganler/code-r1-12k>
* LeetCodeDataset <https://github.com/newfacade/LeetCodeDataset>
* verl 文档：`docs/preparation/reward_function.rst`、`docs/examples/config.rst`
* 奖励分派入口 `verl/utils/reward_score/__init__.py` ｜ 奖励循环 `verl/experimental/reward_loop/`

# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# 最省事的起步方式：官方 12K 数据集已经是 verl 的 RL 格式（data_source="code" +
# reward_model.ground_truth = JSON 测试），直接下载就能训，不必自己造数据。
#
#   python examples/code_r1/data_preprocess/download_code_r1_12k.py --local_dir ./data/code-r1-12k --verify 64
#
# --verify 做的是 Code-R1 反复强调的那一步：**训练前用沙箱重跑参考解**。如果参考解
# 都跑不过，说明奖励链路有问题（沙箱缺依赖/超时太短/数据本身有坏测试），别急着训。

import argparse
import json
import os

from datasets import load_dataset

from verl.utils.reward_score.coder1 import compute_score

# 奖励解析的格式：<think>...</think> 后面跟 <answer>，代码写在 ``` 块里
TRAJECTORY = "<think>\nReference solution.\n</think>\n<answer>\n```python\n{}\n```\n</answer>"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo_id", default="ganler/code-r1-12k")
    parser.add_argument("--local_dir", default="./data/code-r1-12k")
    parser.add_argument("--verify", type=int, default=32, help="用沙箱复跑多少条参考解（0 = 跳过）")
    parser.add_argument("--min_pass_rate", type=float, default=0.0, help="低于这个通过率就报错退出")
    args = parser.parse_args()

    dataset = load_dataset(args.repo_id)
    print(dataset)
    os.makedirs(args.local_dir, exist_ok=True)
    for split in dataset:
        path = os.path.join(args.local_dir, f"{split}.parquet")
        dataset[split].to_parquet(path)
        print(f"wrote {path} ({len(dataset[split])} rows)")

    # 检查字段是否符合奖励函数与数据加载器的预期
    row = dataset["train"][0]
    assert row["data_source"] == "code", row["data_source"]
    assert {"role", "content"} <= set(row["prompt"][0]), "prompt schema changed"
    keys = list(json.loads(row["reward_model"]["ground_truth"]))
    assert set(keys) & {"functional", "pytest", "inputs"}, f"unknown ground truth keys: {keys}"
    print(f"schema OK, ground truth keys = {keys}")

    if args.verify <= 0:
        return

    passed = failed = 0
    for i in range(min(args.verify, len(dataset["train"]))):
        item = dataset["train"][i]
        reference = (item["extra_info"] or {}).get("reference")
        if not reference:
            continue
        score = compute_score(
            solution_str=TRAJECTORY.format(reference),
            ground_truth=item["reward_model"]["ground_truth"],
            extra_info=item["extra_info"],
        )
        passed += score >= 1.0
        failed += score < 1.0
        if score < 1.0:
            print(f"--- sample {i} ({(item['extra_info'] or {}).get('dataset')}) got {score} ---")

    rate = passed / max(passed + failed, 1)
    print(f"verified {passed + failed} references: pass={passed} fail={failed} -> {rate:.1%}")
    if rate < args.min_pass_rate:
        raise SystemExit(
            f"reference pass rate {rate:.1%} < --min_pass_rate {args.min_pass_rate:.1%}; "
            "check CODER1_EXEC (sandbox backend), CODER1_TIMEOUT, and the sandbox's dependencies"
        )


if __name__ == "__main__":
    main()

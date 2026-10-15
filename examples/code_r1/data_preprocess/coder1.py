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
# 造 Code-R1 数据（LeetCode ~2k + 验证过的 TACO ~10k）。
# 移植自 https://github.com/ganler/code-r1 的 examples/data_preprocess/coder1.py
# （上游还有 deepmind/code_contests 和 kodcode 两个数据源，官方 12K 用不到，删掉了）。
#
# Code-R1 的核心观点之一是"数据本身就是奖励函数的一部分"：错测试或不可解的题就是
# 奖励假阳性。所以这个脚本不只是改格式——它**把参考解丢进沙箱跑一遍**，跑不过的样本
# 直接丢掉（26K TACO 最终只剩 10K）。
#
# 用法：
#   git clone https://github.com/newfacade/LeetCodeDataset.git
#   python examples/code_r1/data_preprocess/coder1.py --root_dir ./data --leetcode_dir ./LeetCodeDataset

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

from datasets import concatenate_datasets, load_dataset

from verl.utils.reward_score.coder1 import code_exec, remote_check_stdio

EMPTY = {"data_source": None, "prompt": None, "ability": None, "reward_model": None, "extra_info": None}

# 系统提示词决定了轨迹格式：奖励要求 <think>...</think> ... <answer>...</answer> 结尾，
# 并从 <answer> 里的 ``` 代码块抽代码
SYSTEM_PROMPT = (
    "You are a helpful programming assistant. "
    "The user will ask you a question and you as the assistant solve it. "
    "The assistant first thinks how to solve the task through reasoning and then provides the "
    "user with the final answer. "
    "The reasoning process and answer are enclosed within <think>...</think> and "
    "<answer>...</answer> tags, respectively."
)

PY_IMPORTS = (
    "import heapq\nfrom math import floor, gcd\nimport random\nimport sys\nfrom typing import *\n"
    "from functools import *\nimport collections\nfrom collections import *\nfrom itertools import *\n"
    "from heapq import *\nfrom bisect import *\nfrom string import *\nimport math\nimport datetime\n"
    "inf = float('inf')\n"
)


def build_row(split, prompt, tests_json, idx, dataset, reference):
    """组装 verl RL 数据集的一行（五个字段，见 README 3.2）。"""
    return {
        "data_source": "code",
        "prompt": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
        "ability": "coding",
        "reward_model": {"style": "rule", "ground_truth": tests_json},
        "extra_info": {"split": split, "index": idx, "prompt": prompt, "reference": reference, "dataset": dataset},
    }


def minimize_stdio(inputs, outputs, max_n_tests=8):
    """保留最小的若干条 stdin/stdout 用例（越小跑得越快）。"""
    pairs = []
    for stdin, stdout in zip(inputs, outputs, strict=False):
        stdin = "\n".join(stdin) if isinstance(stdin, list) else stdin
        stdout = "\n".join(stdout) if isinstance(stdout, list) else stdout
        if sys.getsizeof(stdin) <= 4 * 1024:
            pairs.append((stdin, stdout.replace("\r\n", "\n")))
    pairs.sort(key=lambda p: sys.getsizeof(p[0]))
    return [p[0] for p in pairs[:max_n_tests]], [p[1] for p in pairs[:max_n_tests]]


def taco(max_n_tests=8, num_proc=8, n_testset=512):
    """10K 验证过的 TACO 样本。"""
    print("=" * 24, "Loading likaixin/TACO-verified...", "=" * 24)
    dataset = load_dataset("likaixin/TACO-verified")["train"]

    def make_map_fn(split):
        def process_fn(example, idx):
            try:
                oracle = json.loads(example["input_output"])
            except (TypeError, json.JSONDecodeError):
                return EMPTY
            source = example["source"]

            # 跳过格式差的样本
            if source in ["geeksforgeeks", "leetcode"]:
                return EMPTY
            if len("".join(c for c in example["question"] if c.isalnum())) < 100:  # 描述太短
                return EMPTY
            if "image" in example["question"].lower() or "\n![" in example["question"]:  # 带图
                return EMPTY

            prompt_pieces = [
                "Solve the programming task below in a Python markdown code block.",
                example["question"].strip(),
            ]
            if example["starter_code"].strip():
                starter = example["starter_code"].strip()
                prompt_pieces += [
                    "Also feel free to reuse/extend the following starter code:",
                    f"```python\n{starter}\n```",
                ]

            if "fn_name" in oracle:
                # 把 TACO 的 inputs/outputs 变成一个真正的断言测试，并先用参考解验证它
                fn_name = oracle["fn_name"]
                tests = f"""\
_inputs = {oracle["inputs"]}
_outputs = {oracle["outputs"]}
import math
def _deep_eq(a, b, tol=1e-5):
    if isinstance(a, float) or isinstance(b, float):
        return math.isclose(a, b, rel_tol=tol, abs_tol=tol)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(_deep_eq(x, y, tol) for x, y in zip(a, b))
    return a == b

for i, o in zip(_inputs, _outputs):
"""
                if source in ["leetcode", "hackerrank"]:
                    tests += f"    assert _deep_eq({fn_name}(*i), o)"
                elif source == "codewars":
                    tests += f"    assert _deep_eq({fn_name}(*i), o[0])"
                else:
                    raise ValueError(f"Unknown source: {source}")

                check_test = example["solutions"][-1] + "\n" + tests
                if source == "leetcode":
                    check_test = PY_IMPORTS + check_test
                succ, err = code_exec(check_test)
                if not succ:
                    print(f"[skip] reference solution fails the test ({source}): {err[:200]}")
                    return EMPTY
                tests_json = json.dumps({"functional": tests})
            elif "inputs" in oracle and "outputs" in oracle:
                stdins, stdouts = minimize_stdio(oracle["inputs"], oracle["outputs"], max_n_tests)
                if not stdins:
                    return EMPTY
                with ThreadPoolExecutor(max_workers=min(len(stdins), 8)) as pool:
                    futures = [
                        pool.submit(remote_check_stdio, example["solutions"][-1], i, o)
                        for i, o in zip(stdins, stdouts, strict=False)
                    ]
                    for future in as_completed(futures):
                        succ, output, stdin, expected = future.result()
                        if not succ or output.strip() != expected.strip():
                            print(f"[skip] reference solution fails on stdin={stdin!r}")
                            return EMPTY
                tests_json = json.dumps({"inputs": stdins, "outputs": stdouts})
            else:
                raise ValueError(f"Unknown ground truth format: {oracle}")

            return build_row(
                split, "\n".join(prompt_pieces), tests_json, idx, "likaixin/TACO-verified", example["solutions"][0]
            )

        return process_fn

    dataset = dataset.map(
        make_map_fn("train"), with_indices=True, num_proc=num_proc, remove_columns=dataset.column_names
    ).filter(lambda row: row != EMPTY)
    split = dataset.train_test_split(test_size=max(1, min(n_testset, len(dataset) * 0.1)), seed=666)
    return split["train"], split["test"]


def leetcode(leetcode_dir, n_testset=512):
    """LeetCode 2K（测试最可靠，单位样本的信号最好）。"""
    print("=" * 24, "Loading LeetCodeDataset...", "=" * 24)
    base = os.path.join(leetcode_dir, "data")
    files = {
        "test": "LeetCodeDataset-v2-test-problems.jsonl",
        "train": ["LeetCodeDataset-v2-rl-problems.jsonl", "LeetCodeDataset-v2-sft-problems.jsonl"],
    }
    missing = [os.path.join(base, f) for f in files["test"] if not os.path.exists(os.path.join(base, f))]
    for name in files["train"]:
        if not os.path.exists(os.path.join(base, name)):
            missing.append(os.path.join(base, name))
    if missing:
        raise FileNotFoundError(
            "找不到 LeetCodeDataset 的 jsonl 文件：\n  "
            + "\n  ".join(missing)
            + "\n请先 git clone https://github.com/newfacade/LeetCodeDataset.git，并用 --leetcode_dir 指向它"
        )

    test_dataset = load_dataset("json", data_files=os.path.join(base, files["test"]), split="train")
    train_dataset = concatenate_datasets(
        [load_dataset("json", data_files=os.path.join(base, f), split="train") for f in files["train"]]
    )

    # 去重，并排除测试集里的题目（防止泄漏）
    test_ids = {row["meta"]["question_id"] for row in test_dataset}
    train_dataset = train_dataset.filter(lambda row: row["meta"]["question_id"] not in test_ids)
    seen, keep = set(), []
    for i, row in enumerate(train_dataset):
        if row["meta"]["question_id"] not in seen:
            seen.add(row["meta"]["question_id"])
            keep.append(i)
    train_dataset = train_dataset.select(keep)
    print(f"train={len(train_dataset)} test={len(test_dataset)}")

    def make_map_fn(split):
        def process_fn(example, idx):
            row = example["meta"]
            prompt = (
                "Please solve the programming task below using a self-contained code snippet "
                f"in a markdown code block.\n\n{row['query'].strip()}"
            )
            # `test` 里定义了 check(candidate)，再把 check(entry_point) 追加进去一起执行
            tests_json = json.dumps({"functional": f"{example['test']}\n\ncheck({example['entry_point'].strip()})"})
            return build_row(split, prompt, tests_json, idx, "LeetCodeDataset", example["completion"])

        return process_fn

    train_dataset = train_dataset.map(make_map_fn("train"), with_indices=True)
    test_dataset = test_dataset.map(make_map_fn("test"), with_indices=True).select(range(n_testset))
    return train_dataset, test_dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root_dir", default="./data/")
    parser.add_argument(
        "--leetcode_dir",
        default=os.environ.get("LEETCODE_DATASET_DIR"),
        help="LeetCodeDataset 仓库路径",
    )
    parser.add_argument("--datasets", nargs="+", default=["leetcode", "taco"], choices=["leetcode", "taco"])
    parser.add_argument("--max_n_tests", type=int, default=8, help="每个样本保留多少条 stdin/stdout 用例")
    parser.add_argument("--num_proc", type=int, default=8)
    args = parser.parse_args()

    train_sets, test_sets = [], []
    for name in args.datasets:
        if name == "taco":
            train, test = taco(args.max_n_tests, args.num_proc)
        else:
            train, test = leetcode(args.leetcode_dir)
        train_sets.append(train)
        test_sets.append(test)

    train_dataset, test_dataset = concatenate_datasets(train_sets).shuffle(seed=666), concatenate_datasets(test_sets)
    print("Train:", train_dataset, "\nTest:", test_dataset)

    local_dir = os.path.join(args.root_dir, f"code-r1-{round(len(train_dataset) / 1000)}k-" + "-".join(args.datasets))
    os.makedirs(local_dir, exist_ok=True)
    train_dataset.to_parquet(os.path.join(local_dir, "train.parquet"))
    test_dataset.to_parquet(os.path.join(local_dir, "test.parquet"))
    print(f"saved to {local_dir}")

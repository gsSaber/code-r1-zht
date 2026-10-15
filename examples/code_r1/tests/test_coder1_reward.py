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
# 奖励单测：不需要 GPU、不需要 Ray，本机即可跑。
#   CODER1_EXEC=local pytest -q examples/code_r1/tests/
#
# `tests/resources/code_r1_12k_sample.json` 是官方数据集 ganler/code-r1-12k
# train 第 0 行的原始内容（题目、测试、参考解），所以这里也有真实的端到端覆盖。

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("CODER1_EXEC", "local")
os.environ.setdefault("CODER1_LOG", "0")

_REPO_ROOT = Path(__file__).resolve().parents[3]
# 奖励模块只依赖标准库；没装齐 verl（ray/torch）时直接按路径导入
try:
    from verl.utils.reward_score.coder1 import compute_score
except Exception:  # noqa: BLE001
    sys.path.insert(0, str(_REPO_ROOT / "verl" / "utils" / "reward_score"))
    from coder1 import compute_score

SAMPLE = json.loads((Path(__file__).parent / "resources" / "code_r1_12k_sample.json").read_text(encoding="utf-8"))


def trajectory(code: str) -> str:
    """把代码包成奖励解析器认识的轨迹格式。"""
    return f"<think>\nLet me solve it.\n</think>\n<answer>\n```python\n{code}\n```\n</answer>"


# ---------------------------------------------------------------------------
# coder1 单元测试奖励
# ---------------------------------------------------------------------------
def test_functional_tests_pass_and_fail():
    gt = json.dumps({"functional": "assert add(1, 2) == 3"})
    assert compute_score(trajectory("def add(a, b):\n    return a + b\n"), gt, extra_info={}) == 1.1
    assert compute_score(trajectory("def add(a, b):\n    return a - b\n"), gt, extra_info={}) == 0.1  # 只有格式分


def test_stdio_tests():
    gt = json.dumps({"inputs": ["2 3\n", "10 20\n"], "outputs": ["5\n", "30\n"]})
    good = "a, b = map(int, input().split())\nprint(a + b)"
    bad = "a, b = map(int, input().split())\nprint(a * b)"
    assert compute_score(trajectory(good), gt, extra_info={}) == 1.1
    assert compute_score(trajectory(bad), gt, extra_info={}) == 0.1


def test_bad_format_gets_negative_reward():
    # 没有 </answer> 收尾，或 <answer> 里没有代码块
    assert compute_score("<think>hi</think>\n```python\nprint(1)\n```", "{}", extra_info={}) == -1.1
    assert compute_score("<think>hi</think><answer>只是文字</answer>", "{}", extra_info={}) == -1.1


def test_unusable_ground_truth_scores_zero():
    # 数据没有测试（ground_truth 不是 JSON 测试）：必须记 0 而不是让训练崩掉
    assert compute_score(trajectory("print(1)"), "", extra_info={}) == 0.0
    assert compute_score(trajectory("print(1)"), "not json", extra_info={}) == 0.0
    assert compute_score(trajectory("print(1)"), json.dumps({"unknown": 1}), extra_info={}) == 0.0


# ---------------------------------------------------------------------------
# 官方数据集真实样本（端到端）
# ---------------------------------------------------------------------------
def test_official_sample_reference_solution_passes():
    assert SAMPLE["data_source"] == "code"
    score = compute_score(
        trajectory(SAMPLE["reference"]), SAMPLE["ground_truth"], extra_info={"prompt": SAMPLE["user_prompt"]}
    )
    assert score == 1.1, "官方参考解必须能通过它自己的测试"


def test_official_sample_wrong_solution_only_gets_format_reward():
    wrong = SAMPLE["reference"] + "\n\nassert False, 'deliberately wrong'"
    score = compute_score(trajectory(wrong), SAMPLE["ground_truth"], extra_info={"prompt": SAMPLE["user_prompt"]})
    assert score == 0.1

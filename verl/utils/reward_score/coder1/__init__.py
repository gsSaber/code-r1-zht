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
# Code-R1 奖励：把 <answer> 里的代码丢进沙箱跑单元测试。
#   格式错 / 没有代码块 -> -(1 + 0.1)      （负分，明确惩罚）
#   代码跑挂            -> +0.1           （只给格式分）
#   测试全过            -> +1 + 0.1
# 移植自 https://github.com/ganler/code-r1 (Apache-2.0)。
# 上游还有 docker / ces / kira 三个执行后端，要么需要额外服务、要么官方自己实测
# 假超时率高，这里删掉了，只留 firejail（训练机）和 local（本机调试）。

import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# 沙箱后端：firejail = Linux 真隔离；local = 无隔离，仅用于本地测试/数据校验
CODER1_EXEC = os.environ.get("CODER1_EXEC") or ("local" if os.name == "nt" else "firejail")

if CODER1_EXEC == "firejail":
    from .firejail_exec import code_exec_firejail as code_exec
elif CODER1_EXEC == "local":
    from .local_exec import code_exec_local as code_exec
else:
    raise ValueError(f"Unknown CODER1_EXEC: {CODER1_EXEC}. Use 'firejail' or 'local'.")

_LOG = os.environ.get("CODER1_LOG", "1") not in ("0", "false", "False")
_MAX_CHAR_DISPLAY = 2048


def remote_check_stdio(code, stdin, stdout):
    """跑一条 stdin/stdout 测试（造数据脚本也用它验证参考解）。"""
    succ, output = code_exec(code=code, stdin=stdin)
    return succ, output, stdin, stdout


def validate_response_structure(s) -> bool:
    """Code-R1 要求轨迹以 <think>...</think> ... <answer>...</answer> 结尾。"""
    return bool(re.match(r"<think>.*</think>.*<answer>.*</answer>$", s.strip(), re.DOTALL))


def extract_code_from_string(s) -> str:
    """取最后一个 <answer> 里的所有 ``` 围栏代码块。"""
    answers = re.findall(r"<answer>(.*?)</answer>", s, re.DOTALL)
    answer = answers[-1].strip() if answers else s
    return "\n".join(re.findall(r"```(?:\w+)?\n(.*?)\n```", answer, re.DOTALL)).strip()


def _load_ground_truth(ground_truth) -> dict:
    """ground_truth 是 JSON 字符串，描述测试：{"functional"|"pytest"|"inputs"/"outputs"}。"""
    if isinstance(ground_truth, dict):
        return ground_truth
    try:
        parsed = json.loads(ground_truth)
    except (TypeError, json.JSONDecodeError) as e:
        raise ValueError(f"ground_truth is not valid JSON tests: {ground_truth!r:.128}") from e
    if not isinstance(parsed, dict):
        raise ValueError(f"ground_truth must decode to an object, got {type(parsed)}")
    return parsed


def _compute_score(solution_str, ground_truth, extra_info, format_reward=0.1, answer_reward=1.0):
    log = []

    code = extract_code_from_string(solution_str)
    if not validate_response_structure(solution_str) or not code:
        log += ["-" * 16 + "Bad format detected!" + "-" * 16, solution_str]
        return -answer_reward - format_reward, "\n".join(log)

    tests = _load_ground_truth(ground_truth)
    log.append("-" * 16 + "Extracted Code to Execute" + "-" * 16)
    log.append(code + ("\n" + tests["functional"] if "functional" in tests else ""))
    start = time.time()

    def failed(output=""):
        log.append("!" * 16 + f"⚠️ Test Execution Failed in {time.time() - start:.1f}s" + "!" * 16)
        log.append(str(output)[:_MAX_CHAR_DISPLAY])
        log.append("-" * 16 + "Failed Prompt" + "-" * 16 + "\n" + str(extra_info.get("prompt", "")))
        return format_reward, "\n".join(log)

    output = ""
    if "functional" in tests:
        # LeetCode 风格：测试代码直接追加在解答后面执行
        succ, output = code_exec(code + "\n" + tests["functional"])
        if not succ:
            return failed(output)
    elif "pytest" in tests:
        succ, output = code_exec(code, pytest=tests["pytest"])
        if not succ:
            return failed(output)
    elif "inputs" in tests and "outputs" in tests:
        # stdin/stdout 风格：逐条比对，多条用例并发跑
        stdins, stdouts = tests["inputs"], tests["outputs"]
        with ThreadPoolExecutor(max_workers=min(8, len(stdins))) as pool:
            futures = [pool.submit(remote_check_stdio, code, i, o) for i, o in zip(stdins, stdouts, strict=False)]
            for future in as_completed(futures):
                succ, output, stdin, expected = future.result()
                if not succ or output.strip() != expected.strip():
                    log.append(f"🔎Input: {stdin!r}\n✅Expected: {expected.strip()!r}\n❌Actual: {output.strip()!r}")
                    return failed(output)
    else:
        raise ValueError(f"Unsupported ground truth: {list(tests)}")

    log += ["+" * 16 + "Test Execution Passed!" + "+" * 16, output]
    return format_reward + answer_reward, "\n".join(log)


def compute_score(solution_str, ground_truth, extra_info=None, format_reward=0.1, answer_reward=1.0):
    """`data_source == "code"` 的奖励入口，由 reward_score/__init__.py 调用。"""
    extra_info = extra_info if isinstance(extra_info, dict) else {}
    try:
        score, log = _compute_score(solution_str, ground_truth, extra_info, format_reward, answer_reward)
    except ValueError as e:
        # 数据没有测试（例如判分类数据集）：给 0 并报错，而不是让训练崩掉
        print(f"[code-r1] unusable ground truth -> score 0. {e}")
        return 0.0

    if _LOG:
        marker = "✅" if score == format_reward + answer_reward else "❌"
        print(f"{marker * 8} Reward = {score} {marker * 8}\n{log}\n")
    return score

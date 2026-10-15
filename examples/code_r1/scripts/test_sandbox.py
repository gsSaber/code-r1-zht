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
# 沙箱体检：训练前先确认它真的能用。
#   CODER1_EXEC=firejail python examples/code_r1/scripts/test_sandbox.py
#   CODER1_EXEC=firejail python examples/code_r1/scripts/test_sandbox.py --stress 512 --concurrency 64
#
# --stress 是关键：Code-R1 的论点是"沙箱在负载下假失败 = 奖励报错"，所以要用你打算
# 训练时的并发量压一遍，看有没有失败。

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import cpu_count
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # 源码运行时需要仓库根目录

try:  # verl 装全了就直接用它的包
    from verl.utils.reward_score.coder1 import CODER1_EXEC, code_exec
except Exception:  # noqa: BLE001 - 奖励模块只依赖标准库，可退化成直接导入
    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "verl" / "utils" / "reward_score"))
    from coder1 import CODER1_EXEC, code_exec


def check(note, code, stdin=None, stdout=None, expect_error=False, pytest=None):
    print("=" * 20, note, "=" * 20)
    succ, output = code_exec(code, stdin=stdin, pytest=pytest)
    print(f"{succ = } {output[:200] = }")
    assert not succ if expect_error else succ, output
    if stdout is not None:
        assert output == stdout, f"expecting {stdout!r} but got {output!r}"


def functional_tests():
    check("plain execution", "print('hello world')", stdout="hello world\n")
    check("third-party dependency", "import numpy as np\nprint(np.ones((6,)).sum())", stdout="6.0\n")
    check("stdin", "print(input())", stdin="hello world", stdout="hello world\n")
    check("error path", "print(1/0)", expect_error=True)
    if CODER1_EXEC in ("firejail", "local"):
        check(
            "pytest collection",
            "def add(a, b):\n    return a + b\n",
            pytest="from solution import add\n\n\ndef test_add():\n    assert add(1, 2) == 3\n",
        )
    # LeetCode 风格：测试代码追加在解答后面
    check(
        "LeetCode 追加测试",
        "def two_sum(nums, target):\n"
        "    seen = {}\n"
        "    for i, n in enumerate(nums):\n"
        "        if target - n in seen:\n"
        "            return [seen[target - n], i]\n"
        "        seen[n] = i\n"
        "assert two_sum([2, 7, 11, 15], 9) == [0, 1]\nprint('ok')",
        stdout="ok\n",
    )


def stress(n_jobs, concurrency):
    code, stdin = "import numpy as np\nprint(np.ones((int(input()),)).sum())", "256"
    print(f"stress: {n_jobs} executions, concurrency {concurrency}")
    start, failures = time.time(), 0
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(code_exec, code, stdin) for _ in range(n_jobs)]
        for future in as_completed(futures):
            succ, output = future.result()
            failures += not succ
            if not succ:
                print(f"FAILED: {output[:300]}")
    elapsed = time.time() - start
    print(f"stress: {n_jobs - failures}/{n_jobs} ok in {elapsed:.1f}s ({n_jobs / max(elapsed, 1e-9):.1f} exec/s)")
    if failures:
        raise SystemExit(
            "failures under load -- the false-positive source Code-R1 warns about: "
            "lower the concurrency, raise CODER1_TIMEOUT, or switch backend"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stress", type=int, default=0, help="executions to stress with (0 = functional only)")
    parser.add_argument("--concurrency", type=int, default=None, help="defaults to cpu_count()")
    args = parser.parse_args()

    print(f"CODER1_EXEC = {CODER1_EXEC}")
    if CODER1_EXEC == "local":
        print("WARNING: the local backend is not sandboxed -- only run code you trust")
    functional_tests()
    print("functional tests passed")
    if args.stress:
        stress(args.stress, args.concurrency or cpu_count())

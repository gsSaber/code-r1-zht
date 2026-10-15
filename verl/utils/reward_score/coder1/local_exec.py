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
# 本机执行后端（非沙箱）：官方 code-r1 没有它。加它很实际——firejail 只有 Linux 有，
# 而奖励逻辑、数据校验在 Windows/macOS 开发机上也要能跑。
# **它不做任何隔离**，只适合可信输入（参考解、本地调试）。

import os
import subprocess
import sys
import tempfile

from .utils import _DEFAULT_TIMEOUT_SECONDS, _ERROR_MSG_PREFIX

# 超过这个长度就不走 `python -c`（argv 长度限制），改成写临时文件
CLI_ARG_SIZE_LIMIT = 1024 * 3


def code_exec_local(code, stdin: str = None, timeout=_DEFAULT_TIMEOUT_SECONDS, pytest: str = None):
    env = os.environ.copy()
    env["OPENBLAS_NUM_THREADS"] = "1"
    env.pop("PYTHONPATH", None)  # 别把 verl 仓库带进被测程序的 sys.path
    # 本机测试直接用当前解释器（依赖都在这个环境里）；
    # firejail 里必须用系统 python3，见 firejail_exec.py 的注释。
    python = sys.executable or "python3"

    with tempfile.TemporaryDirectory() as tmpdir:
        if pytest:
            # 模型输出写 solution.py，测试写 test_solution.py，交给 pytest 收集
            with open(os.path.join(tmpdir, "solution.py"), "w", encoding="utf-8") as f:
                f.write(code)
            with open(os.path.join(tmpdir, "test_solution.py"), "w", encoding="utf-8") as f:
                f.write(pytest)
            command = [python, "-m", "pytest", tmpdir]
        elif len(code) < CLI_ARG_SIZE_LIMIT:
            command = [python, "-c", code]
        else:
            script = os.path.join(tmpdir, "solution.py")
            with open(script, "w", encoding="utf-8") as f:
                f.write(code)
            command = [python, script]

        try:
            result = subprocess.run(
                command,
                cwd=tmpdir,
                input=stdin.encode() if stdin else None,
                capture_output=True,
                env=env,
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return False, _ERROR_MSG_PREFIX + f"Timeout after {timeout}s"

    # 统一换行，保证 stdin/stdout 奖励在 Windows 和 Linux 上判定一致
    stdout = result.stdout.decode(errors="replace").replace("\r\n", "\n")
    stderr = result.stderr.decode(errors="replace").strip().replace("\r\n", "\n")
    if result.returncode == 0:
        return True, stdout
    return False, _ERROR_MSG_PREFIX + f"Exit code: {result.returncode}\nSTDOUT:\n{stdout}\n\nSTDERR:\n{stderr}"

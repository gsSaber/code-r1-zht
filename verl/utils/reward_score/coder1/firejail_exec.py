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
# firejail 执行后端（官方 code-r1 试了四种方案后的结论：它最可靠）。
# 上游作者的实测笔记：code_exec_server 快但并发下约 10% 假超时；kira 假超时少了但
# 并发极差；e2b 并发好但要付费且默认不支持 stdin；firejail 兼顾隔离、速度与并发
# ——seccomp 屏蔽 socket，rlimit 限制进程数/文件大小/内存/运行时长。
# 安装：
#   sudo add-apt-repository ppa:deki/firejail && sudo apt-get update
#   sudo apt-get install firejail firejail-profiles

import os
import subprocess
from tempfile import NamedTemporaryFile, TemporaryDirectory

from .utils import _DEFAULT_TIMEOUT_SECONDS, _ERROR_MSG_PREFIX

CLI_ARG_SIZE_LIMIT = 1024 * 3


def code_exec_firejail(code, stdin: str = None, timeout=_DEFAULT_TIMEOUT_SECONDS, pytest: str = None):
    env = os.environ.copy()
    env["OPENBLAS_NUM_THREADS"] = "1"
    env.pop("PYTHONPATH", None)  # 避免导入到错误的东西

    command = [
        "firejail",
        "--private",
        "--quiet",
        "--seccomp=socket",
        "--profile=pip",
        "--rlimit-nproc=32",
        "--rlimit-nofile=32",
        "--rlimit-fsize=2m",  # 限制写文件大小
        "--rlimit-as=4096m",  # 限制内存
        f"--timeout=00:00:{timeout}",
    ]
    cwd = None
    tmpdir, tmpfile = None, None
    try:
        if pytest:
            assert stdin is None, "pytest 模式不支持 stdin"
            tmpdir = TemporaryDirectory()
            with open(os.path.join(tmpdir.name, "solution.py"), "w", encoding="utf-8") as f:
                f.write(code)
            with open(os.path.join(tmpdir.name, "test_solution.py"), "w", encoding="utf-8") as f:
                f.write(pytest)
            command += [f"--whitelist={tmpdir.name}", "python3", "-m", "pytest", tmpdir.name]
            cwd = tmpdir.name
        else:
            # 沙箱里只能用系统 python3：`--private` 会把 /home 换成空目录，
            # 传 conda 环境的 sys.executable 会找不到解释器。
            if len(code) < CLI_ARG_SIZE_LIMIT:
                command += ["python3", "-c", code]
            else:
                tmpfile = NamedTemporaryFile()
                tmpfile.write(code.encode())
                tmpfile.flush()
                command += [f"--whitelist={tmpfile.name}", "python3", tmpfile.name]

        result = subprocess.run(
            command,
            cwd=cwd,
            input=stdin.encode() if stdin else None,
            capture_output=True,
            env=env,
            timeout=timeout + 10,  # firejail 自己有 --timeout，这里只是兜底
            check=False,
        )
    except subprocess.TimeoutExpired:
        return False, _ERROR_MSG_PREFIX + f"Timeout after {timeout}s"
    finally:
        if tmpdir:
            tmpdir.cleanup()
        if tmpfile:
            tmpfile.close()

    stdout = result.stdout.decode(errors="replace")
    stderr = result.stderr.decode(errors="replace").strip()
    if result.returncode == 0:
        return True, stdout
    return False, _ERROR_MSG_PREFIX + f"STDOUT:\n{stdout}\n\nSTDERR:\n{stderr}"

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
# 官方 code-r1 把 prime 奖励管理器从进程池改成线程池、并把并发限制在 cpu_count()，
# 目的只有一个：代码奖励是 I/O 密集的（等沙箱），并发过高会让沙箱假超时，而假超时
# 就是错误奖励。
#
# 新版 verl 里"并行度"已经由 reward loop 提供（`reward.num_workers` 个 Ray actor，
# 每个 worker 内部还有 asyncio + 线程池），所以那处改动本身已经不需要了；但**没有人
# 限制单个 worker 一次起多少个沙箱**。这个文件就是那个限制的现代落点：
# 继承 naive 奖励管理器，只把打分函数换成"信号量 + 失败重试"版本。
#
# 用法（不必改任何核心代码）：
#   reward.reward_manager.source=importlib
#   reward.reward_manager.name=CodeR1RewardManager
#   reward.reward_manager.module.path=$REPO_ROOT/examples/code_r1/code_reward_manager.py
#
# 环境变量：CODE_REWARD_CONCURRENCY（默认 cpu_count // reward.num_workers）、CODE_REWARD_RETRIES

import os
import threading

from verl.experimental.reward_loop.reward_manager import register
from verl.experimental.reward_loop.reward_manager.base import RewardManagerBase
from verl.experimental.reward_loop.reward_manager.naive import NaiveRewardManager
from verl.utils.reward_score import default_compute_score

_semaphore = None
_semaphore_lock = threading.Lock()


def _bounded_retrying(reward_fn, retries: int):
    """限制同时在跑的沙箱数量；打分抛异常（沙箱/网络抖动）时重试。"""

    def wrapper(*args, **kwargs):
        for attempt in range(retries + 1):
            # 信号量加在沙箱调用外面：限制的是并发**执行**数，不是排队样本数
            with _semaphore:
                try:
                    return reward_fn(*args, **kwargs)
                except Exception as e:
                    if attempt == retries:
                        raise
                    print(f"[code_r1] reward attempt {attempt + 1} failed ({type(e).__name__}: {e}); retrying")

    return wrapper


@register("code_r1")
class CodeR1RewardManager(RewardManagerBase):
    """naive 奖励管理器 + 沙箱并发上限 + 瞬时失败重试。"""

    def __init__(self, config, tokenizer, compute_score=None, reward_router_address=None, reward_model_tokenizer=None):
        super().__init__(config, tokenizer, compute_score)

        global _semaphore
        with _semaphore_lock:
            if _semaphore is None:
                num_workers = max(1, int(config.reward.get("num_workers", 1) or 1))
                limit = max(1, int(os.environ.get("CODE_REWARD_CONCURRENCY", (os.cpu_count() or 1) // num_workers)))
                _semaphore = threading.Semaphore(limit)
                print(f"[code_r1] at most {limit} concurrent sandbox executions per reward worker")

        bounded = _bounded_retrying(
            compute_score or default_compute_score, retries=int(os.environ.get("CODE_REWARD_RETRIES", "1"))
        )
        self.compute_score = bounded
        # 解码、奖励组装等逻辑直接复用 naive，不重写
        self._inner = NaiveRewardManager(
            config,
            tokenizer,
            bounded,
            reward_router_address=reward_router_address,
            reward_model_tokenizer=reward_model_tokenizer,
        )

    async def run_single(self, data):
        return await self._inner.run_single(data)

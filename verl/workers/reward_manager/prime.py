# Copyright 2024 PRIME team and/or its affiliates
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

import asyncio
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from multiprocessing import cpu_count
from typing import Any, Callable, Optional

import torch
from transformers import PreTrainedTokenizer

from verl import DataProto
from verl.utils.ray_utils import get_event_loop
from verl.utils.reward_score import default_compute_score
from verl.workers.reward_manager import register
from verl.workers.reward_manager.abstract import AbstractRewardManager


async def single_compute_score(evaluation_func, completion, reference, task, task_extra_info, executor, timeout=600.0):
    loop = get_event_loop()
    try:
        # Ensure process_completion is called properly
        future = loop.run_in_executor(executor, partial(evaluation_func, task, completion, reference, task_extra_info))
        return await asyncio.wait_for(future, timeout=timeout)
    except asyncio.TimeoutError:
        print(f"[Timeout] Task timeout: {completion[:80]}")
        return None  # Default value for timed-out rows
    except Exception as e:
        print(f"[Error] Task failed: {e}, completion: {completion[:80]}")
        return None  # Default value for failed rows


async def parallel_compute_score_async(
    evaluation_func, completions, references, tasks, extra_info=None, num_processes=64
):
    # [Code-R1] Thread pool instead of process pool.
    #
    # A code reward is *I/O bound*: every sample spawns a sandboxed subprocess
    # (or posts to a code-execution server) and then waits. Threads give the same
    # concurrency without pickling the (long) completions across process
    # boundaries, without paying fork cost per sample, and without leaving stray
    # processes behind when a task times out -- each call is already bounded by
    # `asyncio.wait_for` below. Upstream code-r1 made exactly this change in
    # `verl/workers/reward_manager/prime.py`; see recipe/code_r1/README.md.
    if extra_info is None:
        extra_info = [None] * len(tasks)
    scores = []
    with ThreadPoolExecutor(max_workers=num_processes) as executor:
        # Create tasks for all rows
        tasks_async = [
            single_compute_score(evaluation_func, c, r, t, ei, executor, timeout=600.0)
            for c, r, t, ei in zip(completions, references, tasks, extra_info, strict=True)
        ]
        results = await asyncio.gather(*tasks_async, return_exceptions=False)

    # Process results
    for result, completion, reference, task in zip(results, completions, references, tasks, strict=True):
        if isinstance(result, Exception) or result is None:
            # Handle failed or timed-out tasks
            scores.append(0.0)
        elif isinstance(result, dict):
            # Reward functions may return {"score": ..., <extras>}.
            scores.append(float(result.get("score", 0.0)))
        elif isinstance(result, int | float | bool):
            scores.append(float(result))
        else:
            scores.append(float(result[0]))
    return scores


def run_reward_scoring(evaluation_func, completions, references, tasks, extra_info=None, num_processes=64):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(
            parallel_compute_score_async(evaluation_func, completions, references, tasks, extra_info, num_processes)
        )
    finally:
        loop.close()


@register("prime")
class PrimeRewardManager(AbstractRewardManager):
    """
    The Reward Manager used in https://github.com/PRIME-RL/PRIME

    NOTE for anyone porting Code-R1: since the reward loop landed, `PPOTrainer`
    resolves `reward.reward_manager.name` against
    `verl.experimental.reward_loop.reward_manager` (naive / dapo / gdpo / limited /
    remote), so this legacy manager is no longer instantiated by the default
    trainer and "prime" is not a valid value there. The thread-pool change below
    is still the historical Code-R1 patch worth reading; the *current* way to
    bound sandbox concurrency is `reward.num_workers` plus
    `recipe/code_r1/code_reward_manager.py`.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        num_examine: int,
        compute_score: Optional[Callable] = None,
        reward_fn_key: str = "data_source",
    ) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or default_compute_score
        self.reward_fn_key = reward_fn_key

    def verify(self, data):
        """
        verify the batch and save as ``acc`` tensor
        """
        # batched scoring
        prompt_ids = data.batch["prompts"]

        response_ids = data.batch["responses"]
        sequences_str = self.tokenizer.batch_decode(response_ids, skip_special_tokens=True)
        ground_truth = [data_item.non_tensor_batch["reward_model"]["ground_truth"] for data_item in data]
        data_sources = data.non_tensor_batch[self.reward_fn_key]
        extra_info = data.non_tensor_batch.get("extra_info", None)

        assert len(sequences_str) == len(ground_truth) == len(data_sources)
        # [Code-R1] Size the pool by the machine rather than a magic 64: the code
        # reward spawns one sandboxed subprocess per test case, so the useful
        # degree of parallelism is bounded by the CPU count.
        num_processes = max(cpu_count(), 1)
        try:
            scores = run_reward_scoring(
                self.compute_score,
                completions=sequences_str,
                references=ground_truth,
                tasks=data_sources,
                extra_info=extra_info,
                num_processes=num_processes,
            )
        except asyncio.TimeoutError:
            print("[Timeout] Global reward scoring timed out. Setting all as 0.")
            scores = [0.0 for _ in range(len(sequences_str))]
        except Exception as e:
            print(f"[Error] Unexpected error during scoring. Setting all as 0. {e}")
            scores = [0.0 for _ in range(len(sequences_str))]
        data.batch["acc"] = torch.tensor(scores, dtype=torch.float32, device=prompt_ids.device)
        return scores

    def __call__(self, data: DataProto, return_dict: bool = False) -> torch.Tensor | dict[str, Any]:
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        reward_from_rm_scores = self._extract_reward_from_rm_scores(data, return_dict)
        if reward_from_rm_scores is not None:
            return reward_from_rm_scores

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)

        already_print_data_sources = {}

        # batched scoring
        prompt_ids = data.batch["prompts"]
        prompt_length = prompt_ids.shape[-1]

        response_ids = data.batch["responses"]
        valid_response_length = data.batch["attention_mask"][:, prompt_length:].sum(dim=-1)
        sequences_str = self.tokenizer.batch_decode(response_ids, skip_special_tokens=True)
        data_sources = data.non_tensor_batch["data_source"]

        scores = self.verify(data)

        for i in range(len(data)):
            data_source = data_sources[i]
            reward_tensor[i, valid_response_length[i].item() - 1] = scores[i]

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print(sequences_str)

        if return_dict:
            return {"reward_tensor": reward_tensor}
        else:
            return reward_tensor

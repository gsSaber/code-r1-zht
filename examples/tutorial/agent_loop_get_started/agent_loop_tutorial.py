#!/usr/bin/env python
# coding: utf-8

# # Train ReAct agent with code sandbox
# 
# In this tutorial, we will demonstrate how to train a [ReAct](https://arxiv.org/abs/2210.03629) agent to solve math problem with code sandbox.
# 
# The agent works as follows:
# 1. Given a math problem, the agent first query LLM to generate response and tool calls, which are python code to be executed in sandbox.
# 2. If there is a tool call, the agent execute the python code in code sandbox.
# 3. After code execution, the agent get the result from sandbox and append to chat history.
# 4. The agent query LLM again until no tool call or max context length reached.
# 
# 
# <figure>
#   <img src="https://langchain-ai.github.io/langgraph/agents/assets/agent.png" alt="ReAct" width="400">
#   <figcaption style="font-style: italic; color: #666;">
#     source: <a href="https://langchain-ai.github.io/langgraph/agents/overview/" target="_blank">LangGraph</a>
#   </figcaption>
# </figure>

# ## 1. Prerequisite
# 
# To run the examples in this notebook, you need to install the verl package first.
# ```bash
# git clone https://github.com/verl-project/verl
# cd verl
# pip install -e .
# ```

# In[1]:


import asyncio
import sys
import tempfile
import os
import socket
import json

import requests
import ray
import fastapi
import uvicorn
from starlette.requests import Request
from starlette.responses import JSONResponse
from pprint import pprint

import verl

ray.init()
verl_config_dir = os.path.join(os.path.dirname(verl.__file__), "trainer/config")


# For demo purpose, we will use Qwen/Qwen3-1.7B as the LLM. First, let's download required model and dataset used in this tutorial.

# In[ ]:


import pyarrow.parquet as pq
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="verl-team/lighteval-MATH-preprocessed",
    repo_type="dataset",
    local_dir=os.path.expanduser("~/verl-team/lighteval-MATH-preprocessed"),
)
snapshot_download(
    repo_id="Qwen/Qwen3-1.7B",
    repo_type="model",
    local_dir=os.path.expanduser("~/Qwen/Qwen3-1.7B"),
)

model_path = os.path.expanduser("~/Qwen/Qwen3-1.7B")
train_file = os.path.expanduser("~/verl-team/lighteval-MATH-preprocessed/train.parquet")
test_file = os.path.expanduser("~/verl-team/lighteval-MATH-preprocessed/test.parquet")

test = pq.read_table(test_file)
test_file = os.path.expanduser("~/verl-team/lighteval-MATH-preprocessed/test_100.parquet")
pq.write_table(test[:100], test_file)


# verl support both vllm and sglang rollout server for high performance inference. This tutorial has been tested on both vllm and sglang, you can choose either of them to run the tutorial.

# In[3]:


rollout_name = "???"  # vllm or sglang


# ## 2. Basic tool call
# For beginning, let's see how we can do basic tool call in verl with example from [Transformer tool use](https://huggingface.co/docs/transformers/main/chat_extras#tool-use). To use tool in verl, we need to define a tool class that inherits from `BaseTool`, and implement the following methods:
# - `get_openai_tool_schema`: return the schema of the tool in `OpenAIFunctionToolSchema` format.
# - `execute`: execute the tool with the given parameters, and return the result in `ToolResponse` format.

# In[4]:


from transformers.utils import get_json_schema
from verl.tools.base_tool import BaseTool, OpenAIFunctionToolSchema, ToolResponse


class WeatherTool(BaseTool):
    def get_current_temperature(self, location: str, unit: str = "celsius"):
        """Get current temperature at a location.

        Args:
            location: The location to get the temperature for, in the format "City, State, Country".
            unit: The unit to return the temperature in. Defaults to "celsius". (choices: ["celsius", "fahrenheit"])

        Returns:
            the temperature, the location, and the unit in a dict
        """
        return {
            "temperature": 26.1,
            "location": location,
            "unit": unit,
        }

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        schema = get_json_schema(self.get_current_temperature)
        return OpenAIFunctionToolSchema(**schema)

    async def execute(self, instance_id: str, parameters: dict, **kwargs) -> tuple[ToolResponse, float, dict]:
        try:
            result = self.get_current_temperature(**parameters)
            return ToolResponse(text=json.dumps(result)), 0, {}
        except Exception as e:
            return ToolResponse(text=str(e)), 0, {}


weather_tool = WeatherTool(config={}, tool_schema=None)


# Next, let's launch a standalone rollout server without hybrid engine (which is more heavy to start) to test the basic tool call.

# In[ ]:


from hydra import compose, initialize_config_dir
from verl.workers.rollout.replica import get_rollout_replica_class

with initialize_config_dir(config_dir=verl_config_dir):
    config = compose(
        config_name="ppo_trainer",
        overrides=[
            "actor_rollout_ref.rollout.name=" + rollout_name,
            "actor_rollout_ref.rollout.mode=async",
            "actor_rollout_ref.rollout.tensor_model_parallel_size=1",
            "actor_rollout_ref.model.path=" + model_path,
            "actor_rollout_ref.rollout.response_length=4096",
            "actor_rollout_ref.rollout.skip_tokenizer_init=False",
            # standalone server has no trainer to sync weights from
            "actor_rollout_ref.rollout.load_format=auto",
            "+actor_rollout_ref.rollout.engine_kwargs.vllm.enable_auto_tool_choice=True",
            "+actor_rollout_ref.rollout.engine_kwargs.vllm.tool_call_parser=hermes",
            "+actor_rollout_ref.rollout.engine_kwargs.sglang.tool_call_parser=qwen25",
        ],
    )

rollout_server_class = get_rollout_replica_class(config.actor_rollout_ref.rollout.name)
rollout_server = rollout_server_class(
    replica_rank=0,
    config=config.actor_rollout_ref.rollout,
    model_config=config.actor_rollout_ref.model,
)

await rollout_server.init_standalone()


# Then, we can query LLM with openai client. Note that we need to pass the tool schema to server to guide LLM generating tool calls. We can see that the LLM correctly generates a tool call to get the temperature in Paris.

# In[6]:


from openai import AsyncOpenAI

client = AsyncOpenAI(
    api_key="dummy",
    base_url=f"http://{rollout_server._server_address}/v1",
)

messages = [{"role": "user", "content": "Hey, what's the temperature in Paris right now?"}]
completion = await client.chat.completions.create(
    model=config.actor_rollout_ref.model.path,
    messages=messages,
    tools=[weather_tool.tool_schema.model_dump(exclude_unset=True, exclude_none=True)],
    extra_body={
        "chat_template_kwargs": {"enable_thinking": False},
    },
)

message = completion.choices[0].message.model_dump(exclude_unset=True, exclude_none=True)
messages.append(message)
pprint(messages)


# We can execute the tool call with arguments generated by LLM and get the temperature in Paris.

# In[7]:


args = json.loads(message["tool_calls"][0]["function"]["arguments"])
tool_response, _, _ = await weather_tool.execute("", args)
print(tool_response)


# Then, we can add the tool response to chat history and query LLM again. With the tool response, LLM can generate a final response to the user.

# In[8]:


messages.append({"role": "tool", "content": tool_response.text})
completion = await client.chat.completions.create(
    model=config.actor_rollout_ref.model.path,
    messages=messages,
    tools=[weather_tool.tool_schema.model_dump(exclude_unset=True, exclude_none=True)],
    extra_body={
        "chat_template_kwargs": {"enable_thinking": False},
    },
)

message = completion.choices[0].message.model_dump(exclude_unset=True, exclude_none=True)
messages.append(message)
pprint(messages)


# ## 2. Advanced tool call with code sandbox
# 
# Now, let's see a more realistic example of tool call with code sandbox, which is widely used in real-world applications.
# 
# ### 2.1 Implement a naive code sandbox
# 
# To execute python code snippet generated by LLM, we need a code sandbox environment. In this tutorial, we will implement a very naive code sandbox, which is
# a FastAPI http server with `/run_code` endpoint. The server works as follows:
# 1. Receive a http request, write the python code snippet to a temp file.
# 2. Spawn a subprocess to execute the code, and get stdout and stderr of the subprocess.
# 3. Return the stdout and stderr of the subprocess as http response.
# 
# > 🚨 **WARNING:** This naive code sandbox is for demonstration purpose only, do not use it in production. Please use docker/kata container for stronger isolation and security restriction.

# In[9]:


@ray.remote(num_cpus=1)
class Sandbox:
    """Sandbox to execute python code."""

    def __init__(self):
        self.address = ray._private.services.get_node_ip_address()
        self.port = self._get_free_port()
        asyncio.create_task(self._start_fastapi_server())

    async def code_execution(self, request: Request):
        request_json = await request.json()
        code = request_json["code"]
        # print(f"execute code:\n{code}")

        _, temp_file = tempfile.mkstemp(suffix=".py", prefix="temp_code", dir=None, text=True)
        with open(temp_file, "w") as f:
            f.write(code)

        try:
            process = await asyncio.create_subprocess_exec(
                sys.executable, temp_file, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )

            stdout, stderr = await process.communicate()

            response = {
                "status": "Success" if process.returncode == 0 else "Failed",
                "run_result": {
                    "status": "Finished",
                    "stdout": stdout.decode(),
                    "stderr": stderr.decode(),
                    "return_code": process.returncode,
                },
            }
            return JSONResponse(content=response)
        finally:
            try:
                os.unlink(temp_file)
            except Exception:
                pass

    def _get_free_port(self):
        with socket.socket() as sock:
            sock.bind(("", 0))
            return sock.getsockname()[1]

    async def _start_fastapi_server(self):
        app = fastapi.FastAPI()
        app.router.add_api_route("/run_code", self.code_execution, methods=["POST"])

        config = uvicorn.Config(app, host=["::", "0.0.0.0"], port=self.port, log_level="warning")
        server = uvicorn.Server(config)
        await server.serve()

    async def get_server_address(self) -> str:
        """Get FastAPI server address."""
        return f"{self.address}:{self.port}"


# In[10]:


sandbox = Sandbox.remote()
sandbox_address = ray.get(sandbox.get_server_address.remote())


# ### 2.2 Define sandbox tool
# 
# As shown in the previous section, we also defined a tool for the code sandbox. In the `execute` method, we send the code snippet to code sandbox by http request and get the output.

# In[11]:


import re
import aiohttp


class SandboxTool(BaseTool):
    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        # Different model may use different code pattern, e.g. python, py, etc.
        self.code_pattern = re.compile(r"```py(.*?)```", re.DOTALL)

    async def code_interpreter(self, code: str) -> str:
        """Execute the code in the sandbox.

        Args:
            code: The code to be executed.

        Returns:
            str: The output of the code execution.
        """
        async with aiohttp.ClientSession() as session:
            async with session.post(
                self.config.get("sandbox_fusion_url"),
                json={"code": code},
            ) as resp:
                resp.raise_for_status()
                result = await resp.json()
                stdout, stderr = result["run_result"]["stdout"], result["run_result"]["stderr"]
                return stdout + stderr

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        schema = get_json_schema(self.code_interpreter)
        return OpenAIFunctionToolSchema(**schema)

    async def execute(self, instance_id: str, parameters: dict, **kwargs) -> tuple[str, float, dict]:
        code = parameters["code"]
        matches = self.code_pattern.findall(code)
        if matches:
            code = matches[0].strip()

        # NOTE: Some script may not explicitly print result, we need to add a print statement to the end of the script.
        # More better way is to SFT the model to make it print result by default, we skip SFT stage in this tutorial.
        lines = code.split("\n")
        for i, line in reversed(list(enumerate(lines))):
            if line == "":
                continue
            if not lines[i].startswith("print"):
                lines[i] = f"print({line})"
            break
        code = "\n".join(lines)

        result = await self.code_interpreter(code)
        return ToolResponse(text=result), 0.0, {}


sandbox_tool = SandboxTool(config={"sandbox_fusion_url": f"http://{sandbox_address}/run_code"}, tool_schema=None)


# First, let's try to execute a valid code and check the response with stdout.

# In[12]:


code = """```py
import sympy

print(sympy.sqrt(3))
```"""

print(await sandbox_tool.execute(instance_id="", parameters={"code": code}))


# Then, let's try to execute an invalid code and check the response with stderr. The error message is important to inform LLM to fix code in next generation.

# In[13]:


code_invalid = """
print(sympy.sqrt(3))
"""

print(await sandbox_tool.execute(instance_id="", parameters={"code": code_invalid}))


# ### 2.3 Test sandbox tool
# 
# Now, we can test sandbox tool with real math problem. In this tutorial, we will use the [DigitalLearningGmbH/MATH-lighteval](https://huggingface.co/datasets/DigitalLearningGmbH/MATH-lighteval) dataset, which consists of problems from mathematics competitions, including the AMC 10, AMC 12, AIME, and more.

# In[14]:


from datasets import load_dataset

dataset = load_dataset("parquet", data_files=test_file)["train"]


# For debug purpose, we can implement ReAct agent as a simple loop. For RL training, there are more subtle issue and corner case to deal with, we provide a built-in ReAct agent loop which will be discussed in next section.

# In[15]:


messages = dataset["prompt"][0]

while True:
    # 1. Chat with the model
    completion = await client.chat.completions.create(
        model=config.actor_rollout_ref.model.path,
        messages=messages,
        tools=[sandbox_tool.tool_schema.model_dump(exclude_unset=True, exclude_none=True)],
        extra_body={
            "chat_template_kwargs": {"enable_thinking": False},
        },
    )

    message = completion.choices[0].message.model_dump(exclude_unset=True, exclude_none=True)
    messages.append(message)

    # 2. Call tools
    finish_reason = completion.choices[0].finish_reason
    if finish_reason != "tool_calls":
        print(f"No tool calls, finish_reason: {finish_reason}")
        break

    try:
        tool_calls = completion.choices[0].message.tool_calls[0]
        args = json.loads(tool_calls.function.arguments)
        result, _, _ = await sandbox_tool.execute("", args)
    except Exception as e:
        print(f"Error: {e}")

    # 3. Add tool response to messages
    messages.append(
        {
            "role": "tool",
            "content": result.text,
        }
    )


# In[16]:


messages


# We can see that the ReAct agent properly query LLM, execute sandbox tool call, finally generate the answer.

# ## 3. End-to-end training with tool agent loop
# 
# After tool has been implemented and tested, we can do end-to-end RL training to tune the model to properly use the tool. To simplify agentic RL training, verl provide [Agent Loop](https://verl.readthedocs.io/en/latest/advance/agent_loop.html) abstraction, which allow user to define custom agent loop:
# - Search agent
# - Math agent
# - SWE agent
# - GUI agent
# - ...
# 
# For ease of use, verl provide two pre-defined agent loop:
# - SingleTurnAgentLoop: single-turn conversation without tool calling
# - ToolAgentLoop: multi-turn conversation with tool calling, interaction
# 
# To use ToolAgentLoop, user only need to provide tools configuration in json/yaml file. In the configuration file, user should specify following fields for each tool:
# - class_name: fully qualified class name of the tool used to dynamically load the custom tool class
# - config: key-word arguments used to initialize the tool instance
# 
# Let's dump our sandbox tool configuration to a json file:

# In[17]:


ray.shutdown()

sandbox = Sandbox.remote()
sandbox_address = ray.get(sandbox.get_server_address.remote())

tool_config = {
    "tools": [
        {
            "class_name": "sandbox.SandboxTool",
            "config": {
                "type": "native",
                "sandbox_fusion_url": f"http://{sandbox_address}/run_code",
            },
        },
    ],
}

tool_config_path = "tool_config.json"
with open(tool_config_path, "w") as f:
    json.dump(tool_config, f)


# In[18]:


from hydra import compose, initialize_config_dir

with initialize_config_dir(config_dir=verl_config_dir):
    config = compose(
        config_name="ppo_trainer",
        overrides=[
            "algorithm.adv_estimator=grpo",
            "data.train_files=" + train_file,
            "data.val_files=" + test_file,
            "data.return_raw_chat=True",
            "data.train_batch_size=32",
            "data.max_prompt_length=1024",
            "data.max_response_length=1024",
            "+data.apply_chat_template_kwargs.enable_thinking=False",
            # actor related
            "actor_rollout_ref.model.path=" + model_path,
            "actor_rollout_ref.actor.ppo_mini_batch_size=8",
            "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8",
            "actor_rollout_ref.actor.fsdp_config.param_offload=True",
            "actor_rollout_ref.actor.fsdp_config.optimizer_offload=True",
            # rollout related
            "actor_rollout_ref.rollout.name=" + rollout_name,
            "actor_rollout_ref.rollout.mode=async",
            "actor_rollout_ref.rollout.tensor_model_parallel_size=1",
            "actor_rollout_ref.rollout.n=8",
            "actor_rollout_ref.rollout.multi_turn.tool_config_path=" + tool_config_path,
            "actor_rollout_ref.rollout.agent.default_agent_loop=tool_agent",
            "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8",
            # trainer related
            "trainer.val_before_train=True",
            "trainer.log_val_generations=10",
            "trainer.n_gpus_per_node=8",
            "trainer.test_freq=-1",
            "trainer.total_training_steps=5",
            "trainer.logger=['console','tensorboard', 'wandb']",
            "trainer.project_name=verl",
            "trainer.experiment_name=" + os.path.basename(model_path),
        ],
    )


# In[ ]:


from verl.trainer.main_ppo import main

main(config)


# For demo purpose, we only train 5 steps, you can verify the training process by checking wandb metrics:
# - num_turns: min/max/mean chat conversation turns in each step.
# - critic rewards: min/max/mean critic rewards in each step.
# 
# For more realistic agentic RL training, please refer to our recipe:
# - [retool](https://github.com/verl-project/verl-recipe/tree/main/retool): implementation of paper [ReTool: Reinforcement Learning for Strategic Tool Use in LLMs](https://arxiv.org/abs/2504.11536)
# - [collabllm](https://github.com/verl-project/verl-recipe/tree/main/collabllm): implementation of paper [CollabLLM: From Passive Responders to Active Collaborators](https://arxiv.org/pdf/2502.00640)
# - [deepeyes](https://github.com/verl-project/verl-recipe/tree/main/deepeyes): implementation of paper [DeepEyes: Incentivizing "Thinking with Images" via Reinforcement Learning](https://arxiv.org/abs/2505.14362)

#!/usr/bin/env python
# coding: utf-8

# # VeRL Ray API Tutorial

# ## Chapter 1: Ray Basics

# In[144]:


import os


# In[145]:


import warnings

import ray
import torch

warnings.filterwarnings("ignore")


# In[146]:


# Build a local ray cluster. The head node and worker node are on this machine
ray.init()


# Implement an Accumulator class.

# In[147]:


@ray.remote
class Accumulator:
    def __init__(self):
        self.value = 0

    def add(self, x):
        self.value += x

    def get_value(self):
        return self.value


# In[148]:


# Instantiate an accumulator. Accumulator can be viewed as a process, acting as an RPC service.
accumulator = Accumulator.remote()


# In[149]:


value_ref = accumulator.get_value.remote()  # Check the current value. Note that this function returns immediately and does not actually wait for the remote execution to complete.
# Get the value
value = ray.get(value_ref)
print(value)


# In[150]:


# Accumulate, then check the result.
accumulator.add.remote(10)  # Similarly, the 'add' here will return immediately.
new_value = ray.get(accumulator.get_value.remote())
print(new_value)


# ## Chapter 2: Resource Pool and RayWorkerGroup
# In the previous example, it was a simple single-process worker. 
# In this example, we implement a worker with a GPU and form a RayWorkerGroup. Within this RayWorkerGroup, we implement a simple operation of an accumulator.

# In[151]:


from verl.single_controller.base import Worker
from verl.single_controller.ray.base import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup, merge_resource_pool


# In[152]:


resource_pool = RayResourcePool([4], use_gpu=True)


# In[153]:


@ray.remote
class GPUAccumulator(Worker):
    def __init__(self) -> None:
        super().__init__()
        # The initial value of each rank is the same as the rank
        self.value = torch.zeros(size=(1,), device="cuda") + self.rank

    def add(self, x):
        self.value += x
        print(f"rank {self.rank}, value: {self.value}")
        return self.value.cpu()


# In[154]:


# Each worker's initial value is its rank, and then each rank's value is incremented by 1, so the values obtained on each rank are [1, 2, 3, 4]
class_with_args = RayClassWithInitArgs(cls=GPUAccumulator)
worker_group = RayWorkerGroup(resource_pool, class_with_args)
print(worker_group.execute_all_sync("add", x=[1, 1, 1, 1]))


# The principle of parameter passing: The input parameter is a list of length world_size, where each element in the list is dispatched respectively to each worker in the RayWorkerGroup. 
# The return parameter is also a list, corresponding to the return value of each worker.

# ### GPU Resource Sharing

# RayWorkerGroups mapped to the same resource pool share the GPU. In this example, we implement three resource pools: the first occupies 4 GPUs, the second also occupies 4 GPUs, and the last occupies all 8 GPUs. Among them, the first resource pool reuses the resource pool mentioned above.

# In[155]:


# Create a new resource pool and then merge the newly created resource pool with the previous one.
resource_pool_1 = RayResourcePool([4], use_gpu=True, name_prefix="a")
resource_pool_merge = merge_resource_pool(resource_pool, resource_pool_1)


# In[156]:


# Establish a RayWorkerGroup on the newly created resource pool.
worker_group_1 = RayWorkerGroup(resource_pool_1, class_with_args)
worker_group_merge = RayWorkerGroup(resource_pool_merge, class_with_args)


# In[157]:


# Run 'add' on the second set of 4 GPUs; the result should be [2, 3, 4, 5].
output_1 = worker_group_1.execute_all_sync("add", x=[2, 2, 2, 2])
print(output_1)


# In[158]:


# Run 'add' on the merged set of 8 GPUs; the result should be [3, 4, 5, 6, 7, 8, 9, 10].
output_merge = worker_group_merge.execute_all_sync("add", x=[3, 3, 3, 3, 3, 3, 3, 3])
print(output_merge)


# In[159]:


print(worker_group.world_size, worker_group_1.world_size, worker_group_merge.world_size)


# ## Chapter 3: Data Dispatch, Execution and Collection

# In the above example, we used the `execute_all_sync` function in the RayWorkerGroup to dispatch data from the driver to each worker. This is very inconvenient for coding. 
# In this chapter, we use the form of function decorators to allow RayWorkerGroup to directly call functions written in the Worker, and to greatly simplify parameter passing.

# In[160]:


from verl.single_controller.base.decorator import Dispatch, Execute, register


# In[161]:


@ray.remote
class GPUAccumulatorDecorator(Worker):
    def __init__(self) -> None:
        super().__init__()
        # The initial value of each rank is the same as the rank
        self.value = torch.zeros(size=(1,), device="cuda") + self.rank

    # map from a single input to all the worker
    @register(Dispatch.ONE_TO_ALL)
    def add(self, x):
        print(x)
        self.value = self.value + x
        print(f"rank {self.rank}, value: {self.value}")
        return self.value.cpu()


# In[162]:


class_with_args = RayClassWithInitArgs(cls=GPUAccumulatorDecorator)
gpu_accumulator_decorator = RayWorkerGroup(resource_pool_merge, class_with_args)


# In[163]:


# As we can see, 10 is automatically dispatched to each Worker in this RayWorkerGroup.
print(gpu_accumulator_decorator.add(x=10))


# ### Custom Dispatch, Collection
# Users can customize `dispatch` and `collection` function. You only need to write the `dispatch_fn` and `collect_fn` functions yourself. We also support executing RPC only on rank_zero, with specific examples provided below.

# In[164]:


from verl.single_controller.base.decorator import Dispatch, collect_all_to_all, register


# In[ ]:


def two_to_all_dispatch_fn(worker_group, *args, **kwargs):
    """
    Assume the input is a list of 2. Duplicate the input interleaved and pass to each worker.
    """
    for arg in args:
        assert len(arg) == 2
        for i in range(worker_group.world_size - 2):
            arg.append(arg[i % 2])
    for k, v in kwargs.items():
        assert len(v) == 2
        for i in range(worker_group.world_size - 2):
            v.append(v[i % 2])
    return args, kwargs


@ray.remote
class TestActor(Worker):
    # TODO: pass *args and **kwargs is bug prone and not very convincing
    def __init__(self, x) -> None:
        super().__init__()
        self._x = x

    def foo(self, y):
        return self._x + y

    @register(dispatch_mode=Dispatch.ALL_TO_ALL, execute_mode=Execute.RANK_ZERO)
    def foo_rank_zero(self, x, y):
        return self._x + y + x

    @register(dispatch_mode={"dispatch_fn": two_to_all_dispatch_fn, "collect_fn": collect_all_to_all})
    def foo_custom(self, x, y):
        return self._x + y + x


# In[166]:


class_with_args = RayClassWithInitArgs(cls=TestActor, x=2)
worker_group = RayWorkerGroup(resource_pool, class_with_args)


# In[167]:


output_ref = worker_group.foo_custom(x=[1, 2], y=[5, 6])
assert output_ref == [8, 10, 8, 10]

output_ref = worker_group.foo_rank_zero(x=1, y=2)
assert output_ref == 5


# In[168]:


print(gpu_accumulator_decorator.world_size)


# In[169]:


# Shutdown ray cluster
ray.shutdown()


# ## Chapter 4: NVMegatronRayWorkerGroup

# Due to the Ray issue, we can only support max_colocate_count=1 in RayResourcePool for now. 
# This means that each GPU can only have one process.
# We can support max_colocate > 1 when applying this pull request: https://github.com/ray-project/ray/pull/44385

# Therefore, we need to restart the ray and initialize a new resource_pool to demonstrate the **NVMegatronRayWorkerGroup**

# In[ ]:


# Build a local ray cluster. The head node and worker node are on this machine
ray.init()


# Finally, we implement a `NVMegatronRayWorkerGroup`, within which we create a Megatron and then run a tensor parallel (tp) split Llama mlp layer. Here, we use a complex dispatch mode, `Megatron_COMPUTE`. This dispatch mode assumes that user passes the data partitioned by DP dimension. The data is dispatched to all tp/pp ranks within the same dp group, and ultimately only collects output data from tp=0 and the last pp. In this way, for users that only write code on the driver, the Megatron behind the RPC becomes transparent.

# In[171]:


import sys

current_pythonpath = os.environ.get("PYTHONPATH", "")

new_path = "/opt/tiger/Megatron-LM"

new_pythonpath = f"{new_path}:{current_pythonpath}" if current_pythonpath else new_path

os.environ["PYTHONPATH"] = new_pythonpath

print(new_path)
sys.path.append(new_path)

import megatron

print(megatron.__file__)


# In[172]:


from megatron.core import parallel_state as mpu
from omegaconf import OmegaConf

from verl.single_controller.base.decorator import Dispatch, Execute, register
from verl.single_controller.base.megatron.worker import MegatronWorker
from verl.single_controller.ray.base import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup


# In[173]:


resource_pool = RayResourcePool([4], use_gpu=True, max_colocate_count=1)


# In[174]:


@ray.remote
class MLPLayerWorker(MegatronWorker):
    def __init__(self):
        super().__init__()
        rank = int(os.environ["LOCAL_RANK"])
        torch.distributed.init_process_group(backend="nccl")
        torch.cuda.set_device(rank)

        mpu.initialize_model_parallel(
            tensor_model_parallel_size=4,
            pipeline_model_parallel_size=1,
            virtual_pipeline_model_parallel_size=None,
            pipeline_model_parallel_split_rank=None,
            use_sharp=False,
            context_parallel_size=1,
            expert_model_parallel_size=1,
            nccl_communicator_config_path=None,
        )
        from megatron.core import tensor_parallel

        tensor_parallel.model_parallel_cuda_manual_seed(10)

    @register(Dispatch.ONE_TO_ALL)
    def init_model(self, config):
        from omegaconf import OmegaConf

        from verl.models.llama.megatron.layers import ParallelLlamaMLP
        from verl.utils.megatron_utils import init_model_parallel_config

        megatron_config = OmegaConf.create(
            {
                "sequence_parallel": False,
                "param_dtype": "fp32",
                "tensor_model_parallel_size": mpu.get_tensor_model_parallel_world_size(),
                "pipeline_model_parallel_rank": mpu.get_pipeline_model_parallel_rank(),
                "pipeline_model_parallel_size": mpu.get_pipeline_model_parallel_world_size(),
                "virtual_pipeline_model_parallel_rank": mpu.get_virtual_pipeline_model_parallel_rank(),
                "virtual_pipeline_model_parallel_size": mpu.get_virtual_pipeline_model_parallel_world_size(),
            }
        )

        megatron_config = init_model_parallel_config(megatron_config)
        self.parallel_layer = ParallelLlamaMLP(config=config, megatron_config=megatron_config)

    @register(Dispatch.ONE_TO_ALL)
    def get_weights(self):
        output = {}
        for key, val in self.parallel_layer.named_parameters():
            output[key] = val
        return output

    @register(Dispatch.MEGATRON_COMPUTE)
    def run_layer(self, x):
        x = x.to("cuda")
        y = self.parallel_layer(x)
        return y


# In[175]:


layer_cls = RayClassWithInitArgs(cls=MLPLayerWorker)
layer_worker_group = NVMegatronRayWorkerGroup(
    resource_pool=resource_pool,
    ray_cls_with_init=layer_cls,
)


# In[176]:


print(layer_worker_group.world_size, layer_worker_group.tp_size, layer_worker_group.pp_size, layer_worker_group.dp_size)


# In[177]:


ffn_hidden_size = 11008
batch_size = 16
seq_len = 2048
hidden_size = 4096

config = OmegaConf.create(
    {
        "hidden_size": hidden_size,
        "intermediate_size": ffn_hidden_size,
        "hidden_act": "silu",
        "pretraining_tp": 1,
        "tp": layer_worker_group.tp_size,
    }
)


# In[178]:


x = torch.rand(size=(seq_len, batch_size, hidden_size), dtype=torch.float32)


# In[179]:


layer_worker_group.init_model(config)


# In[180]:


output = layer_worker_group.run_layer(
    [x]
)  # This must be a list of size 1, ensuring that the input equals the data parallel (dp).
print(output[0].shape)


# In[181]:


# Shutdown ray cluster
ray.shutdown()


import random
from typing import Optional

import torch

from .manager_modules import (
    ConvLayerMemoryManager,
    LinearLayerMemoryManager,
    _is_quantized_tensor,
)
from .ring_allocator import RingBufferAllocator

LINEAR_MODULES = [
    "Linear",
    "LoRACompatibleLinear",
    "QLinear",
]
CONV_MODULES = [
    "Conv2d",
    "LoRACompatibleConv",
    "QConv2d",
]

UNMANAGED_MODULES = [
    "LayerNorm",
    "BatchNorm1d",
    "BatchNorm2d",
    "BatchNorm3d",
    "GroupNorm",
    "InstanceNorm1d",
    "InstanceNorm2d",
    "InstanceNorm3d",
    "Embedding",
    "EmbeddingBag",
    "RNNBase",
    "LSTM",
    "GRU",
    "RNN",
    "Conv3d"
]

UNMANAGED_MODULES_INCLUDES = ["RotaryEmbedding", "Norm", "RotaryPosEmbed"]


class MemoryManager:
    def __init__(
        self,
        module: torch.nn.Module,
        process_device: torch.device = torch.device("cpu"),
    ):
        self.module: torch.nn.Module = module
        self.process_device: torch.device = process_device
        self.unmanaged_modules: list[torch.nn.Module] = []

    def memory_managed_to(self, *args, **kwargs):
        # first move all the unmanaged modules
        for module in self.unmanaged_modules:
            if isinstance(module, torch.nn.Parameter):
                # Parameter cannot move this way
                module.data = module.data.to(*args, **kwargs)
            else:
                module.to(*args, **kwargs)
        # check for a dtype argument
        dtype = None
        if "dtype" in kwargs:
            dtype = kwargs["dtype"]
        elif len(args) > 0:
            for i, arg in enumerate(args):
                if isinstance(arg, torch.dtype):
                    dtype = arg
                    break
        if dtype is not None:
            return self.module._mm_to(dtype=dtype)
        return self.module

    @staticmethod
    def _estimate_model_size_bytes(module: torch.nn.Module) -> int:
        total = 0
        for p in module.parameters():
            total += p.numel() * p.element_size()
        return total

    def initialize_ring_allocators(
        self,
        model_size_bytes: int,
        device: torch.device,
        gpu_fraction: float = 0.25,
    ) -> None:
        from .manager_modules import _get_device_state

        if device.type != "cuda":
            return

        state = _get_device_state(device)
        target_bytes = max(1, int(model_size_bytes * gpu_fraction))
        existing = state.get("ring_allocator_gpu")
        if existing is not None and getattr(existing, "total_capacity", 0) >= target_bytes:
            state["use_ring_allocator"] = True
            return

        if existing is not None:
            existing.deallocate_cache()

        state["ring_allocator_gpu"] = RingBufferAllocator(device, target_bytes=target_bytes)
        state["use_ring_allocator"] = True

    def disable_ring_allocators(self, device: torch.device) -> None:
        from .manager_modules import _get_device_state

        state = _get_device_state(device)
        allocator = state.get("ring_allocator_gpu")
        if allocator is not None:
            allocator.deallocate_cache()
            state["ring_allocator_gpu"] = None
        state["use_ring_allocator"] = False

    @staticmethod
    def get_or_create_layer_transfer_allocators(
        train_device: torch.device,
        temp_device: torch.device,
        gpu_target_bytes: int,
        cpu_target_bytes: int,
    ) -> tuple[RingBufferAllocator, RingBufferAllocator]:
        from .manager_modules import _get_device_state

        state = _get_device_state(train_device)

        gpu_allocator: Optional[RingBufferAllocator] = state.get("layer_transfer_allocator_gpu")
        if gpu_allocator is None or getattr(gpu_allocator, "total_capacity", 0) < gpu_target_bytes:
            if gpu_allocator is not None:
                gpu_allocator.deallocate_cache()
            gpu_allocator = RingBufferAllocator(train_device, target_bytes=max(1, gpu_target_bytes))
            state["layer_transfer_allocator_gpu"] = gpu_allocator

        cpu_allocator: Optional[RingBufferAllocator] = state.get("layer_transfer_allocator_cpu")
        if cpu_allocator is None or getattr(cpu_allocator, "total_capacity", 0) < cpu_target_bytes:
            if cpu_allocator is not None:
                cpu_allocator.deallocate_cache()
            cpu_allocator = RingBufferAllocator(temp_device, target_bytes=max(1, cpu_target_bytes))
            state["layer_transfer_allocator_cpu"] = cpu_allocator

        return gpu_allocator, cpu_allocator

    @staticmethod
    def release_layer_transfer_allocators(train_device: torch.device) -> None:
        from .manager_modules import _get_device_state

        state = _get_device_state(train_device)
        for key in ("layer_transfer_allocator_gpu", "layer_transfer_allocator_cpu"):
            allocator = state.get(key)
            if allocator is not None:
                allocator.deallocate_cache()
                state[key] = None

    @classmethod
    def attach(
        cls,
        module: torch.nn.Module,
        device: torch.device,
        offload_percent: float = 1.0,
        ignore_modules: list[torch.nn.Module] = [],
        use_ring_allocator: bool = True,
        ring_allocator_gpu_fraction: float = 0.25,
    ):
        if hasattr(module, "_memory_manager"):
            # already attached
            return

        module._memory_manager = cls(module, device)
        if use_ring_allocator and device.type == "cuda":
            model_size_bytes = cls._estimate_model_size_bytes(module)
            module._memory_manager.initialize_ring_allocators(
                model_size_bytes=model_size_bytes,
                device=device,
                gpu_fraction=ring_allocator_gpu_fraction,
            )

        # override the to method to handle memory management
        module._mm_to = module.to
        module.to = module._memory_manager.memory_managed_to

        # add ignore modules to unmanaged list
        for im in ignore_modules:
            module._memory_manager.unmanaged_modules.append(im)

        # count ignore modules as processed
        modules_processed = [x for x in ignore_modules]
        # attach to all modules
        for name, sub_module in module.named_modules():
            for child_name, child_module in sub_module.named_modules():
                if (
                    child_module.__class__.__name__ in LINEAR_MODULES
                    and child_module not in modules_processed
                ):
                    weight = getattr(child_module, "weight", None)
                    if _is_quantized_tensor(weight):
                        # Keep native quantized kernels/dispatch for numerical parity.
                        module._memory_manager.unmanaged_modules.append(child_module)
                        modules_processed.append(child_module)
                        continue
                    skip = False
                    if offload_percent < 1.0:
                        # randomly skip some modules
                        if random.random() > offload_percent:
                            skip = True
                    if skip:
                        module._memory_manager.unmanaged_modules.append(child_module)
                    else:
                        # linear
                        LinearLayerMemoryManager.attach(
                            child_module, module._memory_manager
                        )
                        # attach to ARA as well
                        if hasattr(child_module, "ara_lora_ref"):
                            ara = child_module.ara_lora_ref()
                            if ara not in modules_processed:
                                MemoryManager.attach(
                                    ara,
                                    device,
                                    use_ring_allocator=use_ring_allocator,
                                    ring_allocator_gpu_fraction=ring_allocator_gpu_fraction,
                                )
                    modules_processed.append(child_module)
                elif (
                    child_module.__class__.__name__ in CONV_MODULES
                    and child_module not in modules_processed
                ):
                    weight = getattr(child_module, "weight", None)
                    if _is_quantized_tensor(weight):
                        # Keep native quantized kernels/dispatch for numerical parity.
                        module._memory_manager.unmanaged_modules.append(child_module)
                        modules_processed.append(child_module)
                        continue
                    skip = False
                    if offload_percent < 1.0:
                        # randomly skip some modules
                        if random.random() > offload_percent:
                            skip = True
                    if skip:
                        module._memory_manager.unmanaged_modules.append(child_module)
                    else:
                        # conv
                        ConvLayerMemoryManager.attach(
                            child_module, module._memory_manager
                        )
                        # attach to ARA as well
                        if hasattr(child_module, "ara_lora_ref"):
                            ara = child_module.ara_lora_ref()
                            if ara not in modules_processed:
                                MemoryManager.attach(
                                    ara,
                                    device,
                                    use_ring_allocator=use_ring_allocator,
                                    ring_allocator_gpu_fraction=ring_allocator_gpu_fraction,
                                )
                            modules_processed.append(ara)
                    modules_processed.append(child_module)
                elif child_module.__class__.__name__ in UNMANAGED_MODULES or any(
                    inc in child_module.__class__.__name__
                    for inc in UNMANAGED_MODULES_INCLUDES
                ):
                    # unmanaged
                    module._memory_manager.unmanaged_modules.append(child_module)
                else:
                    continue

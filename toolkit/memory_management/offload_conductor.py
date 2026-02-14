"""
Coordinated layer offloading conductor.

Manages CPU<->GPU transfers for transformer blocks with:
- Pre-computed offload schedules
- Dedicated CUDA streams
- Overlapped compute and transfer
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Optional, Set

import torch
from torch import nn

from .ring_allocator import RingBufferAllocator


def get_layer_bytes(layer: nn.Module) -> int:
    total = 0
    for param in layer.parameters():
        total += param.numel() * param.element_size()
    return total


@dataclass
class OffloadSchedule:
    layers_to_load: list[int]
    layers_to_offload: list[int]


class OffloadStrategy:
    def __init__(self, layers: list[nn.Module], layer_offload_fraction: float):
        self.num_layers = len(layers)
        self.layer_bytes = [get_layer_bytes(layer) for layer in layers]
        self.total_bytes = sum(self.layer_bytes)

        target_loaded_bytes = int(self.total_bytes * (1.0 - layer_offload_fraction))
        self.initial_loaded = self._compute_initial_layers(target_loaded_bytes)

        self.forward_backward_loaded = [
            self._get_layers_below(i, target_loaded_bytes, is_forward=True, is_cyclic=False)
            for i in range(self.num_layers)
        ]
        self.forward_forward_loaded = [
            self._get_layers_below(i, target_loaded_bytes, is_forward=True, is_cyclic=True)
            for i in range(self.num_layers)
        ]
        self.backward_forward_loaded = [
            self._get_layers_below(i, target_loaded_bytes, is_forward=False, is_cyclic=False)
            for i in range(self.num_layers)
        ]

        all_loaded = self.forward_backward_loaded + self.forward_forward_loaded + self.backward_forward_loaded
        self.max_loaded_bytes = max(sum(self.layer_bytes[i] for i in indices) for indices in all_loaded)
        min_loaded_bytes = min(sum(self.layer_bytes[i] for i in indices) for indices in all_loaded)
        self.max_offloaded_bytes = self.total_bytes - min_loaded_bytes + max(self.layer_bytes)

    def _compute_initial_layers(self, target_bytes: int) -> list[int]:
        return self._get_layers_below(0, target_bytes, is_forward=True, is_cyclic=False)

    def _get_layers_below(self, start_layer: int, max_bytes: int, is_forward: bool, is_cyclic: bool) -> list[int]:
        accumulator = 0
        layers: list[int] = []

        if is_forward and is_cyclic:
            for i in range(start_layer, self.num_layers):
                if accumulator + self.layer_bytes[i] > max_bytes and len(layers) >= 2:
                    break
                accumulator += self.layer_bytes[i]
                layers.append(i)
            for i in range(start_layer):
                if accumulator + self.layer_bytes[i] > max_bytes and len(layers) >= 2:
                    break
                accumulator += self.layer_bytes[i]
                layers.append(i)
        elif is_forward and not is_cyclic:
            for i in range(start_layer, self.num_layers):
                if accumulator + self.layer_bytes[i] > max_bytes and len(layers) >= 2:
                    break
                accumulator += self.layer_bytes[i]
                layers.append(i)
            for i in range(start_layer - 1, -1, -1):
                if accumulator + self.layer_bytes[i] > max_bytes and len(layers) >= 2:
                    break
                accumulator += self.layer_bytes[i]
                layers.append(i)
        else:
            for i in range(start_layer, -1, -1):
                if accumulator + self.layer_bytes[i] > max_bytes and len(layers) >= 2:
                    break
                accumulator += self.layer_bytes[i]
                layers.append(i)
            for i in range(start_layer + 1, self.num_layers):
                if accumulator + self.layer_bytes[i] > max_bytes and len(layers) >= 2:
                    break
                accumulator += self.layer_bytes[i]
                layers.append(i)

        return sorted(layers)

    def get_schedule(
        self,
        layer_index: int,
        is_forward: bool,
        is_next_forward: bool,
        loaded_layers: Set[int],
    ) -> OffloadSchedule:
        if is_forward and is_next_forward:
            target = set(self.forward_forward_loaded[layer_index])
        elif is_forward:
            target = set(self.forward_backward_loaded[layer_index])
        else:
            target = set(self.backward_forward_loaded[layer_index])

        to_offload = sorted(loaded_layers - target)
        to_load = sorted(target - loaded_layers)

        if is_forward:
            to_offload = [i for i in to_offload if i >= layer_index] + [i for i in to_offload if i < layer_index]
        else:
            rev = list(reversed(to_offload))
            to_offload = [i for i in rev if i < layer_index] + [i for i in rev if i >= layer_index]

        return OffloadSchedule(layers_to_load=to_load, layers_to_offload=to_offload)


class SyncEvent:
    def __init__(self, event: Optional[torch.cuda.Event] = None):
        self.event = event

    def record(self, stream: Optional[torch.cuda.Stream] = None):
        if self.event is None:
            return
        if stream is not None:
            self.event.record(stream)
        else:
            self.event.record()

    def wait(self, stream: torch.cuda.Stream):
        if self.event is not None:
            stream.wait_event(self.event)

    def synchronize(self):
        if self.event is not None:
            self.event.synchronize()


class OffloadConductor:
    def __init__(
        self,
        train_device: torch.device,
        temp_device: torch.device = torch.device("cpu"),
        layer_offload_fraction: float = 0.5,
        strict_gradient_offload: bool = True,
    ):
        self.train_device = train_device
        self.temp_device = temp_device
        self.layer_offload_fraction = layer_offload_fraction
        self.strict_gradient_offload = strict_gradient_offload

        self.layers: list[nn.Module] = []
        self.layer_device_map: list[Optional[torch.device]] = []
        self.layer_train_events: list[SyncEvent] = []
        self.layer_transfer_events: list[SyncEvent] = []

        self.strategy: Optional[OffloadStrategy] = None
        self.gpu_allocator: Optional[RingBufferAllocator] = None
        self.cpu_allocator: Optional[RingBufferAllocator] = None

        self.is_forward_pass = True
        self.keep_graph = False
        self.is_active = False
        self._deferred_offloads: list[int] = []
        self._recent_events: deque[str] = deque(maxlen=64)
        self._load_count = 0
        self._offload_count = 0
        self._grad_blocked_offload_count = 0
        self._deferred_retry_count = 0
        self._deferred_requeue_count = 0

        if self.train_device.type == "cuda":
            self.train_stream = torch.cuda.default_stream(self.train_device)
            self.layer_transfer_stream = torch.cuda.Stream(self.train_device)
            self.activations_transfer_stream = torch.cuda.Stream(self.train_device)
            self.async_transfer = True
        else:
            self.train_stream = None
            self.layer_transfer_stream = None
            self.activations_transfer_stream = None
            self.async_transfer = False

    def _record_event(self, event: str, layer_index: int, detail: str = "") -> None:
        suffix = f":{detail}" if detail else ""
        self._recent_events.append(f"{event}@{layer_index}{suffix}")

    def add_layer(self, layer: nn.Module) -> int:
        idx = len(self.layers)
        self.layers.append(layer)
        self.layer_device_map.append(None)
        self.layer_train_events.append(SyncEvent())
        self.layer_transfer_events.append(SyncEvent())
        return idx

    def activate(self) -> None:
        if not self.layers:
            raise RuntimeError("No layers registered. Call add_layer() first.")
        if self.is_active:
            return

        self.strategy = OffloadStrategy(self.layers, self.layer_offload_fraction)

        from .manager import MemoryManager

        self.gpu_allocator, self.cpu_allocator = MemoryManager.get_or_create_layer_transfer_allocators(
            train_device=self.train_device,
            temp_device=self.temp_device,
            gpu_target_bytes=self.strategy.max_loaded_bytes,
            cpu_target_bytes=self.strategy.max_offloaded_bytes,
        )

        for i, _layer in enumerate(self.layers):
            if i in self.strategy.initial_loaded:
                self._load_layer_sync(i)
            else:
                self._offload_layer_sync(i)

        if self.async_transfer:
            for i in range(len(self.layers)):
                self.layer_train_events[i] = SyncEvent(torch.cuda.Event())
                self.layer_transfer_events[i] = SyncEvent(torch.cuda.Event())

        self.is_active = True

    def deactivate(self) -> None:
        if not self.is_active:
            return

        if self.async_transfer:
            self._wait_all_transfers()

        for i in range(len(self.layers)):
            if self.layer_device_map[i] == self.train_device:
                self._offload_layer_sync(i)

        from .manager import MemoryManager

        MemoryManager.release_layer_transfer_allocators(self.train_device)

        self.gpu_allocator = None
        self.cpu_allocator = None
        self.is_active = False

    def start_forward(self, keep_graph: bool = True) -> None:
        if not self.is_active:
            return
        if self.async_transfer:
            self.layer_transfer_stream.wait_stream(self.train_stream)
        self._wait_all_transfers()
        self.is_forward_pass = True
        self.keep_graph = keep_graph

    def start_backward(self) -> None:
        if not self.is_active:
            return
        self.is_forward_pass = False

    def before_layer(self, layer_index: int, is_forward_override: Optional[bool] = None) -> None:
        if not self.is_active:
            return

        if is_forward_override is not None:
            self.is_forward_pass = is_forward_override
        elif torch.is_grad_enabled() and self.is_forward_pass:
            # Valid when checkpoint wrappers use `use_reentrant=True`.
            self.is_forward_pass = False

        if self.async_transfer:
            self.layer_transfer_events[layer_index].wait(self.train_stream)

        self._process_deferred_offloads(except_layer=layer_index)

        loaded = self._get_loaded_layers()
        schedule = self.strategy.get_schedule(
            layer_index=layer_index,
            is_forward=self.is_forward_pass,
            is_next_forward=not self.keep_graph,
            loaded_layers=loaded,
        )
        self._record_event(
            "before",
            layer_index,
            f"fwd={int(self.is_forward_pass)},load={len(schedule.layers_to_load)},offload={len(schedule.layers_to_offload)}",
        )

        for idx in schedule.layers_to_offload:
            self._schedule_offload(idx)
        for idx in schedule.layers_to_load:
            self._schedule_load(idx)

    def after_layer(self, layer_index: int) -> None:
        if not self.is_active:
            return
        if self.async_transfer:
            self.layer_train_events[layer_index].record(self.train_stream)

    def _get_loaded_layers(self) -> Set[int]:
        return {i for i, dev in enumerate(self.layer_device_map) if dev == self.train_device}

    def _schedule_load(self, layer_index: int) -> None:
        if self.layer_device_map[layer_index] == self.train_device:
            self._record_event("load_skip", layer_index, "already_on_train")
            return
        self._record_event("load", layer_index)
        if self.async_transfer:
            with torch.cuda.stream(self.layer_transfer_stream):
                self.layer_train_events[layer_index].wait(self.layer_transfer_stream)
                self._load_layer_impl(layer_index)
                self.layer_transfer_events[layer_index].record(self.layer_transfer_stream)
        else:
            self._load_layer_impl(layer_index)

    def _schedule_offload(self, layer_index: int) -> None:
        if self.layer_device_map[layer_index] == self.temp_device:
            self._record_event("offload_skip", layer_index, "already_on_temp")
            return

        layer = self.layers[layer_index]
        has_live_grad = any(param.grad is not None for param in layer.parameters())
        if has_live_grad:
            self._grad_blocked_offload_count += 1
            if self.strict_gradient_offload:
                raise RuntimeError(
                    f"Refusing to offload layer {layer_index} with live gradients while strict offload is enabled."
                )
            if layer_index not in self._deferred_offloads:
                self._deferred_offloads.append(layer_index)
            self._record_event("offload_defer", layer_index, "live_grad")
            return

        self._record_event("offload", layer_index)
        if self.async_transfer:
            with torch.cuda.stream(self.layer_transfer_stream):
                self.layer_train_events[layer_index].wait(self.layer_transfer_stream)
                self._offload_layer_impl(layer_index)
                self.layer_transfer_events[layer_index].record(self.layer_transfer_stream)
        else:
            self._offload_layer_impl(layer_index)

    def _load_layer_impl(self, layer_index: int) -> None:
        if self.gpu_allocator is None or self.cpu_allocator is None:
            raise RuntimeError("OffloadConductor allocators are not initialized.")
        layer = self.layers[layer_index]
        for param in layer.parameters():
            gpu_tensor = self.gpu_allocator.allocate_like(param.data, layer_index)
            if gpu_tensor is None:
                gpu_tensor = param.data.to(self.train_device, non_blocking=self.async_transfer)
            else:
                gpu_tensor.copy_(param.data, non_blocking=self.async_transfer)
            param.data = gpu_tensor
        self.cpu_allocator.deallocate_layer(layer_index)
        self.layer_device_map[layer_index] = self.train_device
        self._load_count += 1

    def _offload_layer_impl(self, layer_index: int) -> None:
        if self.gpu_allocator is None or self.cpu_allocator is None:
            raise RuntimeError("OffloadConductor allocators are not initialized.")
        layer = self.layers[layer_index]
        for param in layer.parameters():
            cpu_tensor = self.cpu_allocator.allocate_like(param.data, layer_index)
            if cpu_tensor is None:
                cpu_tensor = param.data.to(self.temp_device, non_blocking=self.async_transfer)
            else:
                cpu_tensor.copy_(param.data, non_blocking=self.async_transfer)
            param.data = cpu_tensor
        self.gpu_allocator.deallocate_layer(layer_index)
        self.layer_device_map[layer_index] = self.temp_device
        self._offload_count += 1

    def _load_layer_sync(self, layer_index: int) -> None:
        self.layers[layer_index].to(self.train_device)
        self.layer_device_map[layer_index] = self.train_device

    def _offload_layer_sync(self, layer_index: int) -> None:
        self.layers[layer_index].to(self.temp_device)
        self.layer_device_map[layer_index] = self.temp_device

    def _wait_all_transfers(self) -> None:
        for event in self.layer_transfer_events:
            event.synchronize()

    def _process_deferred_offloads(self, except_layer: int) -> None:
        pending = self._deferred_offloads
        self._deferred_offloads = []
        for layer_index in pending:
            if layer_index == except_layer:
                self._deferred_requeue_count += 1
                self._record_event("offload_requeue", layer_index, "current_layer")
                if layer_index not in self._deferred_offloads:
                    self._deferred_offloads.append(layer_index)
                continue
            self._deferred_retry_count += 1
            self._schedule_offload(layer_index)

    def get_stats(self) -> dict[str, Any]:
        loaded = self._get_loaded_layers()
        return {
            "total_layers": len(self.layers),
            "loaded_layers": len(loaded),
            "loaded_indices": sorted(loaded),
            "is_active": self.is_active,
            "is_forward": self.is_forward_pass,
            "deferred_offloads": len(self._deferred_offloads),
            "load_ops": self._load_count,
            "offload_ops": self._offload_count,
            "grad_blocked_offloads": self._grad_blocked_offload_count,
            "deferred_retries": self._deferred_retry_count,
            "deferred_requeues": self._deferred_requeue_count,
            "recent_events": list(self._recent_events),
            "gpu_allocator": self.gpu_allocator.get_stats() if self.gpu_allocator else None,
            "cpu_allocator": self.cpu_allocator.get_stats() if self.cpu_allocator else None,
        }

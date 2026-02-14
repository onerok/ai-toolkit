"""
Static ring buffer allocator for layer weight transfers.

Pre-allocates memory pools to eliminate per-transfer allocation overhead.
"""

from typing import Optional

import torch


def _ceil_16(n: int) -> int:
    """Round up to nearest 16-byte boundary."""
    return n + (16 - (n % 16)) % 16


class RingBufferAllocator:
    """Pre-allocated ring buffer for layer weight transfers."""

    def __init__(
        self,
        device: torch.device,
        target_bytes: int,
        num_cache_tensors: int = 4,
    ):
        self.device = device
        self.is_pinned = device.type == "cpu"

        max_tensor_bytes = max(1, target_bytes // max(1, num_cache_tensors))
        self.cache_tensor_size = _ceil_16(max_tensor_bytes + 4096)
        self.total_capacity = self.cache_tensor_size * max(1, num_cache_tensors)
        self.cache_tensors: list[Optional[torch.Tensor]] = [None] * max(1, num_cache_tensors)

        # Ring tracking in byte space [0, total_capacity)
        self.allocation_start = 0
        self.allocation_end = 0
        self._layer_allocations: dict[int, tuple[int, int]] = {}
        self._pending_deallocations: set[tuple[int, int]] = set()

    def _ensure_tensor(self, index: int) -> None:
        if self.cache_tensors[index] is None:
            tensor = torch.zeros(self.cache_tensor_size, dtype=torch.int8, device=self.device)
            if self.is_pinned:
                tensor = tensor.pin_memory()
            self.cache_tensors[index] = tensor

    def _used_bytes(self) -> int:
        if self.allocation_end >= self.allocation_start:
            return self.allocation_end - self.allocation_start
        return (self.total_capacity - self.allocation_start) + self.allocation_end

    def _split_range(self, start: int, end: int) -> list[tuple[int, int]]:
        if end >= start:
            return [(start, end)]
        return [(start, self.total_capacity), (0, end)]

    def _ranges_overlap(self, a: tuple[int, int], b: tuple[int, int]) -> bool:
        return a[0] < b[1] and b[0] < a[1]

    def _would_overwrite_live(self, new_start: int, new_end: int) -> bool:
        if self._used_bytes() == 0:
            return False

        live_ranges = self._split_range(self.allocation_start, self.allocation_end)
        new_ranges = self._split_range(new_start, new_end)
        for nr in new_ranges:
            for lr in live_ranges:
                if self._ranges_overlap(nr, lr):
                    return True
        return False

    def allocate_like(self, source_tensor: torch.Tensor, layer_index: int = -1) -> Optional[torch.Tensor]:
        raw_num_bytes = source_tensor.numel() * source_tensor.element_size()
        num_bytes = _ceil_16(raw_num_bytes)
        if num_bytes > self.cache_tensor_size:
            return None

        # Keep one-byte sentinel to avoid full/empty ambiguity.
        if self._used_bytes() + num_bytes >= self.total_capacity:
            return None

        cache_idx = self.allocation_end // self.cache_tensor_size
        offset = _ceil_16(self.allocation_end % self.cache_tensor_size)
        if offset + num_bytes > self.cache_tensor_size:
            cache_idx = (cache_idx + 1) % len(self.cache_tensors)
            offset = 0

        new_position = cache_idx * self.cache_tensor_size + offset
        new_end = new_position + num_bytes
        if new_end > self.total_capacity:
            new_position = 0
            new_end = num_bytes
            cache_idx = 0
            offset = 0

        if self._would_overwrite_live(new_position, new_end):
            return None

        self._ensure_tensor(cache_idx)
        byte_view = self.cache_tensors[cache_idx][offset : offset + num_bytes]
        payload_view = byte_view[:raw_num_bytes]
        tensor_view = payload_view.view(dtype=source_tensor.dtype).view(source_tensor.shape)

        self.allocation_end = new_end % self.total_capacity
        if layer_index >= 0:
            self._layer_allocations[layer_index] = (new_position, new_end % self.total_capacity)

        return tensor_view

    def deallocate_layer(self, layer_index: int) -> None:
        if layer_index not in self._layer_allocations:
            return

        start, end = self._layer_allocations.pop(layer_index)
        if start == self.allocation_start:
            self.allocation_start = end
            self._consume_pending_deallocations()
        else:
            self._pending_deallocations.add((start, end))

    def _consume_pending_deallocations(self) -> None:
        changed = True
        while changed and self._pending_deallocations:
            changed = False
            for start, end in list(self._pending_deallocations):
                if start == self.allocation_start:
                    self.allocation_start = end
                    self._pending_deallocations.discard((start, end))
                    changed = True
                    break

    def clear(self) -> None:
        self.allocation_start = 0
        self.allocation_end = 0
        self._layer_allocations.clear()
        self._pending_deallocations.clear()

    def deallocate_cache(self) -> None:
        self.cache_tensors = [None] * len(self.cache_tensors)
        self.clear()

    def get_stats(self) -> dict:
        allocated_tensors = sum(1 for t in self.cache_tensors if t is not None)
        total_bytes = allocated_tensors * self.cache_tensor_size
        used_bytes = self._used_bytes()
        return {
            "device": str(self.device),
            "cache_tensors_allocated": allocated_tensors,
            "cache_tensors_total": len(self.cache_tensors),
            "total_bytes": total_bytes,
            "used_bytes": used_bytes,
            "utilization": used_bytes / total_bytes if total_bytes > 0 else 0.0,
            "active_layers": len(self._layer_allocations),
            "pending_deallocations": len(self._pending_deallocations),
        }

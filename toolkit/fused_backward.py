from __future__ import annotations

from typing import Dict, Iterable, Optional

import torch


class FusedBackwardManager:
    def __init__(self, optimizer, enabled: bool):
        self.optimizer = optimizer
        self._raw_optimizer = self._unwrap_optimizer(optimizer)
        self.enabled_requested = bool(enabled)
        self.enabled = False
        self.disable_reason: Optional[str] = None

        self._param_to_group: Dict[torch.nn.Parameter, dict] = {}
        self._hook_handles = []

        self._is_update_step = False
        self._allow_step_on_backward = False
        self._did_any_param_step = False

        self.stats = {
            "hooks_called": 0,
            "params_stepped": 0,
            "step_parameter_errors": 0,
            "grads_cleared": 0,
            "update_steps_seen": 0,
            "update_steps_fused": 0,
        }

    @staticmethod
    def _unwrap_optimizer(optimizer):
        if hasattr(optimizer, "optimizer"):
            return optimizer.optimizer
        return optimizer

    def supports_fused_step(self) -> bool:
        return hasattr(self._raw_optimizer, "step_parameter")

    def enable(self) -> bool:
        if not self.enabled_requested:
            self.disable_reason = "disabled in config"
            self.enabled = False
            return False
        if not self.supports_fused_step():
            self.disable_reason = "optimizer does not expose step_parameter"
            self.enabled = False
            return False
        self.enabled = True
        self.disable_reason = None
        return True

    def disable(self, reason: str):
        self.enabled = False
        self.disable_reason = reason

    def attach(self, params: Iterable[torch.nn.Parameter]):
        if not self.enabled:
            return

        self.detach()
        self._index_param_groups()

        for param in params:
            if not isinstance(param, torch.nn.Parameter):
                continue
            if not param.requires_grad:
                continue
            handle = param.register_post_accumulate_grad_hook(self._post_accumulate_hook)
            self._hook_handles.append(handle)

    def detach(self):
        for handle in self._hook_handles:
            try:
                handle.remove()
            except Exception:
                pass
        self._hook_handles = []
        self._param_to_group = {}
        self._is_update_step = False
        self._allow_step_on_backward = False
        self._did_any_param_step = False

    def _index_param_groups(self):
        self._param_to_group = {}
        if not hasattr(self._raw_optimizer, "param_groups"):
            return
        for group in self._raw_optimizer.param_groups:
            for param in group.get("params", []):
                if isinstance(param, torch.nn.Parameter):
                    self._param_to_group[param] = group

    def set_update_step(self, is_update_step: bool):
        self._is_update_step = bool(is_update_step)
        self._allow_step_on_backward = False
        self._did_any_param_step = False
        if self._is_update_step:
            self.stats["update_steps_seen"] += 1
            if self.enabled and hasattr(self._raw_optimizer, "begin_fused_update"):
                self._raw_optimizer.begin_fused_update()

    def set_backward_mode(self, allow_step: bool):
        self._allow_step_on_backward = bool(allow_step)

    def did_fused_step_this_update(self) -> bool:
        return self._did_any_param_step

    def _post_accumulate_hook(self, param: torch.nn.Parameter):
        self.stats["hooks_called"] += 1

        if not self.enabled:
            return
        if not self._is_update_step:
            return
        if not self._allow_step_on_backward:
            return
        if not param.requires_grad:
            return

        if param.grad is None and hasattr(param, "_accum_grad"):
            param.grad = param._accum_grad
            del param._accum_grad

        if param.grad is None:
            return

        group = self._param_to_group.get(param, None)
        try:
            self._raw_optimizer.step_parameter(param, group)
            self._did_any_param_step = True
            self.stats["params_stepped"] += 1
        except Exception:
            self.stats["step_parameter_errors"] += 1
            raise

        param.grad = None
        self.stats["grads_cleared"] += 1

    def mark_update_complete(self):
        if self._did_any_param_step:
            self.stats["update_steps_fused"] += 1
        if self.enabled and hasattr(self._raw_optimizer, "end_fused_update"):
            self._raw_optimizer.end_fused_update()

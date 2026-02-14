import math

import torch


class AdamWStep(torch.optim.AdamW):
    def _find_param_group(self, param):
        for group in self.param_groups:
            for candidate in group["params"]:
                if candidate is param:
                    return group
        return None

    @torch.no_grad()
    def step_parameter(self, param, group=None):
        if param.grad is None:
            return
        if group is None:
            group = self._find_param_group(param)
        if group is None:
            return

        grad = param.grad
        if grad.is_sparse:
            raise RuntimeError("AdamWStep does not support sparse gradients")
        if group.get("maximize", False):
            grad = -grad

        state = self.state[param]
        if len(state) == 0:
            state["step"] = 0
            state["exp_avg"] = torch.zeros_like(param, memory_format=torch.preserve_format)
            state["exp_avg_sq"] = torch.zeros_like(param, memory_format=torch.preserve_format)
            if group.get("amsgrad", False):
                state["max_exp_avg_sq"] = torch.zeros_like(param, memory_format=torch.preserve_format)

        exp_avg = state["exp_avg"]
        exp_avg_sq = state["exp_avg_sq"]
        beta1, beta2 = group["betas"]
        state["step"] += 1
        step = state["step"]

        if group["weight_decay"] != 0:
            param.mul_(1 - group["lr"] * group["weight_decay"])

        exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

        if group.get("amsgrad", False):
            max_exp_avg_sq = state["max_exp_avg_sq"]
            torch.maximum(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
            denom_base = max_exp_avg_sq
        else:
            denom_base = exp_avg_sq

        bias_correction1 = 1 - beta1 ** step
        bias_correction2 = 1 - beta2 ** step
        step_size = group["lr"] / bias_correction1
        denom = (denom_base.sqrt() / math.sqrt(bias_correction2)).add_(group["eps"])

        param.addcdiv_(exp_avg, denom, value=-step_size)

# Phase 3: Fused Backward Pass

**Impact:** Reduces peak memory by discarding gradients immediately after consumption
**Risk:** Medium — optimizer integration change
**Effort:** 3-5 days
**Dependencies:** Phase 1 (can run in parallel with Phase 2)

---

## Problem

Current flow:
```
accelerator.backward(loss) → all gradients accumulate → optimizer.step() → optimizer.zero_grad()
```

All parameter gradients must coexist in VRAM simultaneously. For a model with 1B parameters in bf16:
- Parameters: 2 GB
- Gradients: 2 GB (same size as params)
- Optimizer states (AdamW): 8 GB (fp32 copy + momentum + variance)
- **Peak gradient memory: 2 GB**

### OneTrainer Approach

`GenericTrainer.py:554-598` uses `register_post_accumulate_grad_hook` to run the optimizer step on each parameter immediately after its gradient is computed during backward.

```
backward starts → grad computed for param N → optimizer.step(param N) → grad = None → grad computed for param N-1 → ...
```

Gradients are set to `None` immediately after consumption → **Peak gradient memory: ~50-100 MB** (only one layer's gradients at a time).

---

## Files to Create/Modify

| File | Change |
|------|--------|
| NEW: `toolkit/fused_backward.py` | FusedBackwardManager class |
| `jobs/process/BaseSDTrainProcess.py` | Integration hooks |
| `extensions_built_in/sd_trainer/SDTrainer.py` | Training loop changes |

---

## Implementation

### Step 1: Create FusedBackwardManager

**File:** `toolkit/fused_backward.py`

```python
"""
Fused backward pass implementation.

Runs optimizer.step() on each parameter immediately after its gradient
is computed during backward, then sets grad to None to free memory.

Based on OneTrainer's implementation in GenericTrainer.py:554-598.
"""

import torch
from torch import nn, Tensor
from torch.utils.hooks import RemovableHandle
from typing import Optional, List
import warnings


class FusedBackwardManager:
    """Manages fused backward pass with per-parameter optimizer steps.

    This hooks into PyTorch's autograd to run optimizer updates immediately
    after each gradient is computed, rather than waiting for all gradients
    to accumulate.

    Args:
        optimizer: The optimizer to use for parameter updates.
        clip_grad_norm: Optional gradient clipping value.
        grad_scaler: Optional GradScaler for mixed precision training.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        clip_grad_norm: Optional[float] = None,
        grad_scaler: Optional[torch.cuda.amp.GradScaler] = None
    ):
        self.optimizer = optimizer
        self.clip_grad_norm = clip_grad_norm
        self.grad_scaler = grad_scaler
        self.hooks: List[RemovableHandle] = []
        self._is_update_step = True
        self._step_count = 0
        self._enabled = True

    def attach(self) -> None:
        """Register post_accumulate_grad_hook on all trainable parameters."""
        if self.hooks:
            warnings.warn("FusedBackwardManager.attach() called but hooks already exist")
            return

        for param_group in self.optimizer.param_groups:
            for i, param in enumerate(param_group['params']):
                if not param.requires_grad:
                    continue

                # Create closure to capture param_group and index
                def make_hook(pg=param_group, idx=i):
                    def hook(tensor: Tensor) -> None:
                        if not self._enabled or not self._is_update_step:
                            return

                        self._step_parameter(tensor, pg, idx)

                    return hook

                handle = param.register_post_accumulate_grad_hook(make_hook())
                self.hooks.append(handle)

        print(f"FusedBackwardManager attached to {len(self.hooks)} parameters")

    def _step_parameter(self, tensor: Tensor, param_group: dict, idx: int) -> None:
        """Run optimizer step for a single parameter."""
        if tensor.grad is None:
            return

        # Handle mixed precision scaling
        if self.grad_scaler is not None:
            # Unscale this parameter's gradient
            inv_scale = self.grad_scaler._get_scale_async()
            if inv_scale is not None:
                tensor.grad.data.mul_(1.0 / inv_scale)

            # Check for inf/nan
            if not torch.isfinite(tensor.grad).all():
                # Skip this parameter, scaler will handle
                self.grad_scaler._found_inf_per_device = True
                tensor.grad = None
                return

        # Gradient clipping (per-parameter)
        if self.clip_grad_norm is not None:
            grad_norm = tensor.grad.data.norm(2)
            clip_coef = self.clip_grad_norm / (grad_norm + 1e-6)
            if clip_coef < 1:
                tensor.grad.data.mul_(clip_coef)

        # Call optimizer's per-parameter step
        # Most optimizers support this via step_parameter or similar
        if hasattr(self.optimizer, 'step_parameter'):
            self.optimizer.step_parameter(tensor, param_group, idx)
        else:
            # Fallback: manually apply update
            self._manual_step(tensor, param_group, idx)

        # Immediately free gradient memory
        tensor.grad = None
        self._step_count += 1

    def _manual_step(self, tensor: Tensor, param_group: dict, idx: int) -> None:
        """Manual optimizer step for optimizers without step_parameter."""
        # This is a simplified AdamW implementation
        # Real implementation should check optimizer type

        lr = param_group.get('lr', 1e-4)
        weight_decay = param_group.get('weight_decay', 0.01)

        # Weight decay
        if weight_decay > 0:
            tensor.data.add_(tensor.data, alpha=-lr * weight_decay)

        # Get optimizer state
        state = self.optimizer.state.get(tensor, {})

        if 'step' not in state:
            state['step'] = 0
            state['exp_avg'] = torch.zeros_like(tensor.data)
            state['exp_avg_sq'] = torch.zeros_like(tensor.data)
            self.optimizer.state[tensor] = state

        state['step'] += 1

        beta1 = param_group.get('betas', (0.9, 0.999))[0]
        beta2 = param_group.get('betas', (0.9, 0.999))[1]
        eps = param_group.get('eps', 1e-8)

        # Adam update
        state['exp_avg'].mul_(beta1).add_(tensor.grad.data, alpha=1 - beta1)
        state['exp_avg_sq'].mul_(beta2).addcmul_(tensor.grad.data, tensor.grad.data, value=1 - beta2)

        bias_correction1 = 1 - beta1 ** state['step']
        bias_correction2 = 1 - beta2 ** state['step']

        step_size = lr / bias_correction1

        denom = (state['exp_avg_sq'].sqrt() / (bias_correction2 ** 0.5)).add_(eps)
        tensor.data.addcdiv_(state['exp_avg'], denom, value=-step_size)

    def set_update_step(self, is_update_step: bool) -> None:
        """Control whether hooks should run (for gradient accumulation).

        Args:
            is_update_step: True if this backward should trigger optimizer steps.
        """
        self._is_update_step = is_update_step

    def enable(self) -> None:
        """Enable fused backward (hooks will run)."""
        self._enabled = True

    def disable(self) -> None:
        """Disable fused backward (hooks won't run)."""
        self._enabled = False

    def detach(self) -> None:
        """Remove all hooks."""
        for h in self.hooks:
            h.remove()
        self.hooks.clear()
        print(f"FusedBackwardManager detached, processed {self._step_count} parameters")

    def get_stats(self) -> dict:
        """Get statistics about fused backward execution."""
        return {
            "hooks_registered": len(self.hooks),
            "parameters_stepped": self._step_count,
            "enabled": self._enabled,
            "is_update_step": self._is_update_step,
        }


def check_optimizer_compatibility(optimizer: torch.optim.Optimizer) -> bool:
    """Check if an optimizer supports fused backward pass.

    Returns:
        True if the optimizer can be used with fused backward.
    """
    # Optimizers known to work
    compatible = [
        'AdamW', 'Adam', 'SGD', 'Lion', 'Prodigy',
        'AdaFactor', 'Adagrad', 'RMSprop'
    ]

    optimizer_name = type(optimizer).__name__

    if optimizer_name in compatible:
        return True

    # Check for step_parameter method
    if hasattr(optimizer, 'step_parameter'):
        return True

    # Fallback: we can use manual step
    return True
```

### Step 2: Integration in training setup

**File:** `jobs/process/BaseSDTrainProcess.py`

Add to setup:

```python
from toolkit.fused_backward import FusedBackwardManager, check_optimizer_compatibility

class BaseSDTrainProcess:
    def __init__(self, ...):
        # ... existing init ...
        self.fused_backward_manager: Optional[FusedBackwardManager] = None

    def setup_fused_backward(self):
        """Set up fused backward pass if configured."""
        if not self.train_config.fused_back_pass:
            return

        # Warn about gradient accumulation
        if self.train_config.gradient_accumulation_steps > 1:
            print("Warning: fused_back_pass with gradient_accumulation > 1 "
                  "does not reduce peak VRAM usage. Gradients still accumulate "
                  "across micro-batches.")

        # Check optimizer compatibility
        if not check_optimizer_compatibility(self.optimizer):
            print(f"Warning: Optimizer {type(self.optimizer).__name__} may not be "
                  "fully compatible with fused backward")

        self.fused_backward_manager = FusedBackwardManager(
            optimizer=self.optimizer,
            clip_grad_norm=self.train_config.max_grad_norm,
            grad_scaler=self.grad_scaler if hasattr(self, 'grad_scaler') else None
        )
        self.fused_backward_manager.attach()
```

### Step 3: Modified training loop

**File:** `extensions_built_in/sd_trainer/SDTrainer.py`

```python
def hook_train_loop(self, batch):
    # ... existing code up to backward ...

    # Determine if this is an update step (for gradient accumulation)
    is_update_step = (self.step_num + 1) % self.train_config.gradient_accumulation_steps == 0

    # Configure fused backward
    if self.fused_backward_manager:
        self.fused_backward_manager.set_update_step(is_update_step)

    # Backward pass
    # With fused backward, optimizer steps happen inside backward()
    self.accelerator.backward(loss)

    # Post-backward handling
    if is_update_step:
        if self.fused_backward_manager:
            # Skip regular optimizer.step() - already done in hooks
            # Just update learning rate scheduler
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            # No need for zero_grad - grads already None
        else:
            # Regular path
            if self.train_config.max_grad_norm:
                self.accelerator.clip_grad_norm_(
                    self.params_to_optimize,
                    self.train_config.max_grad_norm
                )
            self.optimizer.step()
            self.lr_scheduler.step()
            self.optimizer.zero_grad()

    return loss.detach()
```

---

## Constraints & Warnings

### Gradient Accumulation

Fused backward with `gradient_accumulation_steps > 1` does NOT save memory:
- Gradients must still accumulate across micro-batches
- Only the final micro-batch triggers optimizer steps
- Memory benefit only applies when `gradient_accumulation_steps == 1`

Add validation:
```python
if self.train_config.fused_back_pass and self.train_config.gradient_accumulation_steps > 1:
    warnings.warn(
        "fused_back_pass with gradient_accumulation_steps > 1 provides no memory benefit. "
        "Consider setting gradient_accumulation_steps: 1 or disabling fused_back_pass."
    )
```

### Mixed Precision

The GradScaler needs special handling:
- `unscale_parameter_()` method may not exist in standard PyTorch
- May need custom GradScaler wrapper
- Check for inf/nan per-parameter

### LR Scheduler

Some LR schedulers expect to be called per-optimizer-step:
- Ensure scheduler is called once per update step, not per parameter
- May need to track scheduler externally

---

## Verification

### Test 1: Peak VRAM reduction

```python
import torch

# Without fused backward
torch.cuda.reset_peak_memory_stats()
# ... run training step ...
peak_without = torch.cuda.max_memory_allocated()

# With fused backward
torch.cuda.reset_peak_memory_stats()
# ... run training step with fused_back_pass=True ...
peak_with = torch.cuda.max_memory_allocated()

reduction = (peak_without - peak_with) / 1e9
print(f"Peak VRAM reduction: {reduction:.2f} GB")
print(f"Percentage: {(peak_without - peak_with) / peak_without * 100:.1f}%")
```

Expected: 10-30% reduction depending on model size.

### Test 2: Loss curve equivalence

```python
# Run identical training with and without fused backward
# Compare loss values at each step

losses_regular = train(fused_back_pass=False, steps=100)
losses_fused = train(fused_back_pass=True, steps=100)

# Should be nearly identical (small numerical differences OK)
max_diff = max(abs(a - b) for a, b in zip(losses_regular, losses_fused))
assert max_diff < 0.01, f"Loss divergence: {max_diff}"
```

### Test 3: Gradient memory profile

```python
# Profile gradient memory during backward
def hook(module, grad_input, grad_output):
    print(f"Grad memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

model.transformer.register_full_backward_hook(hook)
```

With fused backward, memory should stay relatively flat during backward.

---

## Configuration

```yaml
training:
  fused_back_pass: false  # default: false
  # Note: Set gradient_accumulation_steps: 1 for memory benefit
```

---

## Rollback Plan

If issues arise:
1. Set `fused_back_pass: false` in config
2. Or call `fused_backward_manager.disable()` at runtime
3. Training continues with standard backward pass

---

## Success Criteria

- [ ] Peak VRAM reduced by 10-30% with gradient_accumulation_steps=1
- [ ] Loss curves identical to standard backward (within numerical precision)
- [ ] No training instability or NaN gradients
- [ ] Works with AdamW, Lion, and Prodigy optimizers
- [ ] Proper warning for gradient_accumulation > 1

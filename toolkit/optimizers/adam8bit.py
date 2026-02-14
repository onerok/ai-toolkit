import math
import copy
import torch
from torch.optim import Optimizer
from toolkit.optimizers.optimizer_utils import copy_stochastic, Auto8bitTensor, stochastic_grad_accummulation

class Adam8bit(Optimizer):
    """
    Implements Adam optimizer with 8-bit state storage and stochastic rounding.
    
    Arguments:
        params (iterable): Iterable of parameters to optimize or dicts defining parameter groups
        lr (float): Learning rate (default: 1e-3)
        betas (tuple): Coefficients for computing running averages of gradient and its square (default: (0.9, 0.999))
        eps (float): Term added to denominator to improve numerical stability (default: 1e-8)
        weight_decay (float): Weight decay coefficient (default: 0)
        decouple (bool): Use AdamW style decoupled weight decay (default: True)
    """
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, 
                 weight_decay=0, decouple=True):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                       decouple=decouple)
        super(Adam8bit, self).__init__(params, defaults)
        
        self.is_stochastic_rounding_accumulation = False
        
        # Setup stochastic grad accumulation hooks
        for group in self.param_groups:
            for param in group['params']:
                if param.requires_grad and param.dtype != torch.float32:
                    self.is_stochastic_rounding_accumulation = True
                    param.register_post_accumulate_grad_hook(
                        stochastic_grad_accummulation
                    )

    @property
    def supports_memory_efficient_fp16(self):
        return False

    @property
    def supports_flat_params(self):
        return True

    def step_hook(self):
        if not self.is_stochastic_rounding_accumulation:
            return
        # Copy over stochastically rounded grads
        for group in self.param_groups:
            for param in group['params']:
                if param.requires_grad and hasattr(param, "_accum_grad"):
                    param.grad = param._accum_grad
                    del param._accum_grad

    def _find_param_group(self, param):
        for group in self.param_groups:
            for candidate in group['params']:
                if candidate is param:
                    return group
        return None

    @torch.no_grad()
    def step_parameter(self, p, group=None):
        if p.grad is None:
            return
        if group is None:
            group = self._find_param_group(p)
        if group is None:
            return

        beta1, beta2 = group['betas']
        eps = group['eps']
        lr = group['lr']
        decay = group['weight_decay']
        decouple = group['decouple']

        grad = p.grad.data.to(torch.float32)
        p_fp32 = p.clone().to(torch.float32)

        if decay != 0 and not decouple:
            grad.add_(p_fp32.data, alpha=decay)

        state = self.state[p]
        if len(state) == 0:
            state['step'] = 0
            state['exp_avg'] = Auto8bitTensor(torch.zeros_like(p_fp32.data).detach())
            state['exp_avg_sq'] = Auto8bitTensor(torch.zeros_like(p_fp32.data).detach())

        exp_avg = state['exp_avg'].to(torch.float32)
        exp_avg_sq = state['exp_avg_sq'].to(torch.float32)

        state['step'] += 1
        bias_correction1 = 1 - beta1 ** state['step']
        bias_correction2 = 1 - beta2 ** state['step']

        exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

        if decay != 0 and decouple:
            p_fp32.data.mul_(1 - lr * decay)

        step_size = lr / bias_correction1
        denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)
        p_fp32.data.addcdiv_(exp_avg, denom, value=-step_size)

        state['exp_avg'] = Auto8bitTensor(exp_avg)
        state['exp_avg_sq'] = Auto8bitTensor(exp_avg_sq)
        copy_stochastic(p.data, p_fp32.data)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.
        
        Arguments:
            closure (callable, optional): A closure that reevaluates the model and returns the loss.
        """
        # Call pre step
        self.step_hook()
        
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                self.step_parameter(p, group=group)

        return loss
    
    def state_dict(self):
        """Returns the state of the optimizer as a dict."""
        # `super().state_dict()` can contain references into internal optimizer state.
        # Deep-copy before converting `Auto8bitTensor` wrappers so training state is untouched.
        state_dict = copy.deepcopy(super().state_dict())
        
        # Convert Auto8bitTensor objects to regular state dicts
        for param_id, param_state in state_dict['state'].items():
            for key, value in param_state.items():
                if isinstance(value, Auto8bitTensor):
                    param_state[key] = {
                        '_type': 'Auto8bitTensor',
                        'state': value.state_dict()
                    }
        
        return state_dict

    def load_state_dict(self, state_dict):
        """Loads the optimizer state."""
        # First, load the basic state
        super().load_state_dict(state_dict)
        
        # Then convert any Auto8bitTensor states back to objects
        for param_id, param_state in self.state.items():
            for key, value in param_state.items():
                if isinstance(value, dict) and value.get('_type') == 'Auto8bitTensor':
                    param_state[key] = Auto8bitTensor(value['state'])

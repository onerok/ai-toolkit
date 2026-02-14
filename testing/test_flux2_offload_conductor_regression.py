import os
import sys

import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from extensions_built_in.diffusion_models.flux2.src.model import Flux2, Flux2Params


def _small_flux2_model() -> Flux2:
    params = Flux2Params(
        in_channels=4,
        context_in_dim=8,
        hidden_size=8,
        num_heads=2,
        depth=0,
        depth_single_blocks=0,
        axes_dim=[2, 2],
        theta=100,
        use_guidance_embed=False,
    )
    return Flux2(params)


def test_configure_offload_conductor_keeps_non_reentrant_checkpointing():
    model = _small_flux2_model()
    model.configure_offload_conductor(train_device=torch.device("cpu"), temp_device=torch.device("meta"))
    assert model._checkpoint_use_reentrant is False


def test_offload_conductor_backward_transition_uses_output_grad_hook():
    class DummyConductor:
        def __init__(self):
            self.is_active = True
            self.forward_calls = 0
            self.backward_calls = 0

        def start_forward(self, keep_graph: bool = True) -> None:
            self.forward_calls += 1

        def start_backward(self) -> None:
            self.backward_calls += 1

    model = _small_flux2_model()
    model.offload_conductor = DummyConductor()
    model.train()

    batch = 2
    img_tokens = 3
    txt_tokens = 2
    x = torch.randn(batch, img_tokens, 4)
    x_ids = torch.randn(batch, img_tokens, 2)
    timesteps = torch.randn(batch)
    ctx = torch.randn(batch, txt_tokens, 8)
    ctx_ids = torch.randn(batch, txt_tokens, 2)

    output = model(x, x_ids, timesteps, ctx, ctx_ids, guidance=None)
    loss = output.square().mean()
    loss.backward()

    assert model.offload_conductor.forward_calls == 1
    assert model.offload_conductor.backward_calls == 1

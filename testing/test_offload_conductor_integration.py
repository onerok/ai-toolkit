import os
import sys
import importlib
from types import SimpleNamespace

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

base_sd_module = importlib.import_module("jobs.process.BaseSDTrainProcess")
from jobs.process.BaseSDTrainProcess import BaseSDTrainProcess


class DummyUnet:
    def __init__(self, activate_return: bool):
        self.activate_return = activate_return
        self.activate_calls = 0

    def activate_offload_conductor(self):
        self.activate_calls += 1
        return self.activate_return


def _make_process(unet: DummyUnet, fused_enabled: bool) -> BaseSDTrainProcess:
    process = BaseSDTrainProcess.__new__(BaseSDTrainProcess)
    process.offload_conductor_enabled = False
    process.model_config = SimpleNamespace(use_offload_conductor=True)
    process.train_config = SimpleNamespace(
        gradient_checkpointing=True,
        fused_back_pass=True,
    )
    process.fused_backward_manager = SimpleNamespace(enabled=fused_enabled)
    process.sd = SimpleNamespace(unet=unet)
    return process


def test_setup_offload_conductor_requires_active_fused_backward(monkeypatch):
    monkeypatch.setattr(base_sd_module, "unwrap_model", lambda module: module)
    unet = DummyUnet(activate_return=True)
    process = _make_process(unet=unet, fused_enabled=False)

    process.setup_offload_conductor()

    assert process.offload_conductor_enabled is False
    assert unet.activate_calls == 0


def test_setup_offload_conductor_does_not_false_positive_when_not_configured(monkeypatch):
    monkeypatch.setattr(base_sd_module, "unwrap_model", lambda module: module)
    unet = DummyUnet(activate_return=False)
    process = _make_process(unet=unet, fused_enabled=True)

    process.setup_offload_conductor()

    assert process.offload_conductor_enabled is False
    assert unet.activate_calls == 1


def test_setup_offload_conductor_enables_when_model_reports_active(monkeypatch):
    monkeypatch.setattr(base_sd_module, "unwrap_model", lambda module: module)
    unet = DummyUnet(activate_return=True)
    process = _make_process(unet=unet, fused_enabled=True)

    process.setup_offload_conductor()

    assert process.offload_conductor_enabled is True
    assert unet.activate_calls == 1

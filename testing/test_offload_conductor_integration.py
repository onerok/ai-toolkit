import os
import sys
import importlib
import json
from types import SimpleNamespace

import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

base_sd_module = importlib.import_module("jobs.process.BaseSDTrainProcess")
from jobs.process.BaseSDTrainProcess import BaseSDTrainProcess
from toolkit.memory_management.offload_conductor import OffloadConductor


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
    process._resolve_model_capability = (
        lambda key, default=False: True if key == "supports_offload_conductor" else default
    )
    return process


def test_setup_offload_conductor_no_longer_requires_fused_backward(monkeypatch):
    monkeypatch.setattr(base_sd_module, "unwrap_model", lambda module: module)
    unet = DummyUnet(activate_return=True)
    process = _make_process(unet=unet, fused_enabled=False)

    process.setup_offload_conductor()

    assert process.offload_conductor_enabled is True
    assert unet.activate_calls == 1


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


def test_offload_conductor_keeps_deferred_current_layer():
    layer = torch.nn.Linear(4, 4)
    conductor = OffloadConductor(
        train_device=torch.device("cpu"),
        temp_device=torch.device("meta"),
        layer_offload_fraction=0.5,
        strict_gradient_offload=False,
    )
    layer_index = conductor.add_layer(layer)
    conductor.layer_device_map[layer_index] = conductor.train_device
    layer.weight.grad = torch.ones_like(layer.weight)

    conductor._schedule_offload(layer_index)
    assert conductor._deferred_offloads == [layer_index]

    conductor._process_deferred_offloads(except_layer=layer_index)
    assert conductor._deferred_offloads == [layer_index]

    stats = conductor.get_stats()
    assert stats["deferred_requeues"] >= 1


def test_write_runtime_knobs_metadata_includes_active_states(tmp_path):
    process = BaseSDTrainProcess.__new__(BaseSDTrainProcess)
    process.accelerator = SimpleNamespace(is_main_process=True)
    process.save_root = str(tmp_path)
    process._requested_runtime_knobs = {
        "model": {"use_offload_conductor": True, "layer_offloading": True},
        "train": {"fused_back_pass": False},
    }
    process._runtime_adjustments = []
    process.offload_conductor_enabled = True
    process.model_config = SimpleNamespace(
        arch="flux",
        quantize=True,
        quantize_te=False,
        qtype="int8",
        use_offload_conductor=True,
        layer_offloading=True,
        layer_offloading_transformer_percent=0.5,
        layer_offloading_text_encoder_percent=0.0,
        low_vram=False,
    )
    process.train_config = SimpleNamespace(
        fused_back_pass=False,
        stable_loss_enabled=False,
        gradient_checkpointing=True,
        optimizer="adamw",
        batch_size=1,
    )
    process.sd = SimpleNamespace(
        model_config=process.model_config,
        get_runtime_adjustments=lambda: [],
    )

    process.write_runtime_knobs_metadata()

    output_path = tmp_path / "runtime_knobs.json"
    assert output_path.exists()
    data = json.loads(output_path.read_text(encoding="utf-8"))
    assert data["active"]["offload_conductor_active"] is True
    assert data["active"]["layer_offloading_active"] is True

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.analyze_phase4_ablations import _check_expected_feature_activation


def test_check_expected_feature_activation_fails_when_expected_but_inactive():
    failures: list[str] = []
    warnings: list[str] = []
    runtime_knobs = {
        "effective": {"model": {"use_offload_conductor": True}},
        "active": {"offload_conductor_active": False},
    }

    _check_expected_feature_activation("offload_conductor", runtime_knobs, failures, warnings)

    assert failures
    assert "offload conductor active" in failures[0]
    assert not warnings


def test_check_expected_feature_activation_warns_when_missing_telemetry():
    failures: list[str] = []
    warnings: list[str] = []
    runtime_knobs = {
        "effective": {"model": {"use_offload_conductor": True}},
    }

    _check_expected_feature_activation("offload_conductor", runtime_knobs, failures, warnings)

    assert warnings
    assert "telemetry is missing" in warnings[0]
    assert not failures

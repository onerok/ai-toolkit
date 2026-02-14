#!/usr/bin/env python3
"""
Performance Benchmark Script for AI Toolkit

This is the core test harness for measuring training performance metrics.
Used by collect_baselines.py to capture reproducible measurements.

Usage:
    # Quick benchmark (20 steps, 256px)
    uv run python scripts/perf_benchmark.py --steps 20 --resolution 256

    # Full benchmark with save cycle
    uv run python scripts/perf_benchmark.py --steps 30 --save-at 15

    # Memory-only test (allocation patterns, no real training)
    uv run python scripts/perf_benchmark.py --memory-only

    # Custom config
    uv run python scripts/perf_benchmark.py --config config/perf_test.yaml
"""

import argparse
import gc
import json
import re
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, ClassVar, Dict, List, Optional

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402


@dataclass
class HardwareInfo:
    """Hardware configuration for reproducibility."""
    gpu: str = ""
    vram_gb: float = 0.0
    cuda_version: str = ""
    driver_version: str = ""
    torch_version: str = ""
    python_version: str = ""

    @classmethod
    def capture(cls) -> "HardwareInfo":
        info = cls()
        info.torch_version = torch.__version__
        info.python_version = sys.version.split()[0]

        if torch.cuda.is_available():
            info.gpu = torch.cuda.get_device_name(0)
            info.vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            info.cuda_version = torch.version.cuda or ""

            # Get driver version from nvidia-smi
            try:
                result = subprocess.run(
                    ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                    capture_output=True, text=True, timeout=5
                )
                if result.returncode == 0:
                    info.driver_version = result.stdout.strip().split("\n")[0]
            except Exception:
                pass

        return info


@dataclass
class MemorySnapshot:
    """CUDA memory state at a point in time."""
    _last_timestamp: ClassVar[float] = 0.0
    timestamp: float
    label: str
    allocated_mb: float = 0.0
    reserved_mb: float = 0.0
    max_allocated_mb: float = 0.0
    num_alloc_retries: int = 0

    @classmethod
    def capture(cls, label: str) -> "MemorySnapshot":
        timestamp = time.time()
        if timestamp <= cls._last_timestamp:
            timestamp = cls._last_timestamp + 1e-6
        cls._last_timestamp = timestamp
        snap = cls(timestamp=timestamp, label=label)
        if torch.cuda.is_available():
            snap.allocated_mb = torch.cuda.memory_allocated() / (1024**2)
            snap.reserved_mb = torch.cuda.memory_reserved() / (1024**2)
            snap.max_allocated_mb = torch.cuda.max_memory_allocated() / (1024**2)
            stats = torch.cuda.memory_stats()
            snap.num_alloc_retries = stats.get("num_alloc_retries", 0)
        return snap


@dataclass
class StepTiming:
    """Timing for a single training step."""
    step: int
    duration_s: float
    is_save_step: bool = False
    is_post_save: bool = False


@dataclass
class BenchmarkMetrics:
    """All metrics captured during a benchmark run."""
    # Timing
    step_times: List[StepTiming] = field(default_factory=list)
    total_duration_s: float = 0.0

    # Memory snapshots at key points
    memory_snapshots: List[MemorySnapshot] = field(default_factory=list)

    # Computed summary metrics
    avg_step_time_s: float = 0.0
    pre_save_step_time_s: float = 0.0
    post_save_step_time_s: float = 0.0
    slowdown_ratio: float = 1.0
    peak_vram_mb: float = 0.0
    pre_save_vram_mb: float = 0.0
    save_peak_vram_mb: float = 0.0
    post_save_vram_mb: float = 0.0
    leaked_mb: float = 0.0
    save_duration_s: float = 0.0
    num_alloc_retries: int = 0
    fragmentation_ratio: float = 0.0
    output_validation_passed: bool = False
    output_validation_message: str = ""
    validated_output_path: str = ""
    generation_validation_passed: bool = False
    generation_validation_message: str = ""
    generation_validation_image_path: str = ""
    generation_clipscore: float = 0.0
    generation_clipscore_threshold: float = 0.0
    generation_clipscore_model: str = ""

    def compute_summary(self, save_at_step: Optional[int] = None):
        """Compute summary metrics from raw data."""
        # Timing metrics (only if we have step data)
        if self.step_times:
            # Average step time (all steps)
            all_times = [s.duration_s for s in self.step_times]
            self.avg_step_time_s = sum(all_times) / len(all_times)

            # Post-save metrics
            if save_at_step is not None:
                pre_save = [s.duration_s for s in self.step_times if s.step < save_at_step]
                post_save = [s.duration_s for s in self.step_times if s.is_post_save]

                if pre_save and post_save:
                    pre_avg = sum(pre_save) / len(pre_save)
                    post_avg = sum(post_save) / len(post_save)
                    self.pre_save_step_time_s = pre_avg
                    self.post_save_step_time_s = post_avg
                    self.slowdown_ratio = post_avg / pre_avg if pre_avg > 0 else 1.0

        # Memory metrics (compute regardless of step data)
        if self.memory_snapshots:
            self.peak_vram_mb = max(s.reserved_mb for s in self.memory_snapshots)
            self.num_alloc_retries = max(s.num_alloc_retries for s in self.memory_snapshots)

            # Find pre-save and post-save snapshots
            pre_save_snap = next((s for s in self.memory_snapshots if s.label == "pre_save"), None)
            save_peak_snap = next((s for s in self.memory_snapshots if s.label == "during_save_peak"), None)
            post_save_snap = next((s for s in self.memory_snapshots if s.label == "post_save_settled"), None)
            if post_save_snap is None:
                post_save_snap = next((s for s in self.memory_snapshots if s.label == "post_save"), None)

            if pre_save_snap:
                self.pre_save_vram_mb = pre_save_snap.reserved_mb
            if save_peak_snap:
                self.save_peak_vram_mb = save_peak_snap.reserved_mb
            if pre_save_snap and post_save_snap:
                self.post_save_vram_mb = post_save_snap.reserved_mb
                self.leaked_mb = post_save_snap.reserved_mb - pre_save_snap.reserved_mb

            # Fragmentation (reserved vs allocated)
            final_snap = self.memory_snapshots[-1] if self.memory_snapshots else None
            if final_snap and final_snap.reserved_mb > 0:
                self.fragmentation_ratio = 1.0 - (final_snap.allocated_mb / final_snap.reserved_mb)


@dataclass
class BenchmarkResult:
    """Complete benchmark result with metadata."""
    tag: str = ""
    commit: str = ""
    timestamp: str = ""
    hardware: HardwareInfo = field(default_factory=HardwareInfo)
    config: Dict[str, Any] = field(default_factory=dict)
    metrics: BenchmarkMetrics = field(default_factory=BenchmarkMetrics)

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict."""
        return {
            "tag": self.tag,
            "commit": self.commit,
            "timestamp": self.timestamp,
            "hardware": asdict(self.hardware),
            "config": self.config,
            "metrics": {
                "peak_vram_mb": self.metrics.peak_vram_mb,
                "pre_save_vram_mb": self.metrics.pre_save_vram_mb,
                "save_peak_vram_mb": self.metrics.save_peak_vram_mb,
                "post_save_vram_mb": self.metrics.post_save_vram_mb,
                "leaked_mb": self.metrics.leaked_mb,
                "avg_step_time_s": self.metrics.avg_step_time_s,
                "pre_save_step_time_s": self.metrics.pre_save_step_time_s,
                "post_save_step_time_s": self.metrics.post_save_step_time_s,
                "slowdown_ratio": self.metrics.slowdown_ratio,
                "save_duration_s": self.metrics.save_duration_s,
                "num_alloc_retries": self.metrics.num_alloc_retries,
                "fragmentation_ratio": self.metrics.fragmentation_ratio,
                "output_validation_passed": self.metrics.output_validation_passed,
                "output_validation_message": self.metrics.output_validation_message,
                "validated_output_path": self.metrics.validated_output_path,
                "generation_validation_passed": self.metrics.generation_validation_passed,
                "generation_validation_message": self.metrics.generation_validation_message,
                "generation_validation_image_path": self.metrics.generation_validation_image_path,
                "generation_clipscore": self.metrics.generation_clipscore,
                "generation_clipscore_threshold": self.metrics.generation_clipscore_threshold,
                "generation_clipscore_model": self.metrics.generation_clipscore_model,
                "total_duration_s": self.metrics.total_duration_s,
                "step_count": len(self.metrics.step_times),
            },
            "raw": {
                "step_times": [asdict(s) for s in self.metrics.step_times],
                "memory_snapshots": [asdict(s) for s in self.metrics.memory_snapshots],
            }
        }


def _get_train_like_processes(job) -> List[Any]:
    """Return processes that expose train loop hooks used for instrumentation."""
    processes = []
    for process in getattr(job, "process", []):
        if all(hasattr(process, attr) for attr in ("save", "end_step_hook", "timer")):
            processes.append(process)
    return processes


def _instrument_training_processes(job, metrics: BenchmarkMetrics, save_at: Optional[int]) -> List[Callable[[], None]]:
    """
    Wrap process hooks to collect per-step timing and save-boundary memory snapshots.
    """
    teardowns: List[Callable[[], None]] = []
    tracked_processes = _get_train_like_processes(job)
    if not tracked_processes:
        return teardowns

    process = tracked_processes[0]
    save_state = {
        "save_step": None,
        "captured_step1": False,
        "captured_step2": False,
        "pending_train_loop_step": None,
        "pending_train_loop_started_monotonic": None,
        "pending_train_loop_timing": None,
    }

    original_timer_start = process.timer.start
    original_timer_stop = process.timer.stop
    original_save = process.save
    original_end_step_hook = process.end_step_hook

    def wrapped_timer_start(timer_name):
        if timer_name == "train_loop":
            save_state["pending_train_loop_step"] = int(getattr(process, "step_num", -1))
            save_state["pending_train_loop_started_monotonic"] = time.perf_counter()
        return original_timer_start(timer_name)

    def wrapped_timer_stop(timer_name):
        if timer_name == "train_loop":
            started_monotonic = save_state.get("pending_train_loop_started_monotonic")
            step_id = save_state.get("pending_train_loop_step", int(getattr(process, "step_num", -1)))
            if started_monotonic is not None:
                elapsed = max(0.0, time.perf_counter() - started_monotonic)
                save_state["pending_train_loop_timing"] = (step_id, elapsed)
            save_state["pending_train_loop_started_monotonic"] = None
            save_state["pending_train_loop_step"] = None
        return original_timer_stop(timer_name)

    def wrapped_save(step=None):
        capture_target_save = save_at is not None and step == save_at
        if not capture_target_save:
            return original_save(step)

        if torch.cuda.is_available():
            metrics.memory_snapshots.append(MemorySnapshot.capture("pre_save"))
            torch.cuda.reset_peak_memory_stats()

        save_started_at = time.time()
        result = original_save(step)
        save_duration = time.time() - save_started_at
        metrics.save_duration_s = max(metrics.save_duration_s, save_duration)

        if save_state["save_step"] is None:
            save_state["save_step"] = int(step)

        if torch.cuda.is_available():
            peak_reserved_mb = torch.cuda.max_memory_reserved() / (1024**2)
            peak_allocated_mb = torch.cuda.max_memory_allocated() / (1024**2)
            peak_snap = MemorySnapshot.capture("during_save_peak")
            peak_snap.reserved_mb = peak_reserved_mb
            peak_snap.allocated_mb = peak_allocated_mb
            peak_snap.max_allocated_mb = peak_allocated_mb
            metrics.memory_snapshots.append(peak_snap)
            metrics.memory_snapshots.append(MemorySnapshot.capture("post_save"))

        return result

    def wrapped_end_step_hook():
        pending_timing = save_state.pop("pending_train_loop_timing", None)
        if pending_timing is not None:
            step_id, elapsed = pending_timing
            save_step = save_state["save_step"]
            metrics.step_times.append(
                StepTiming(
                    step=step_id,
                    duration_s=elapsed,
                    is_save_step=(save_step is not None and step_id == save_step),
                    is_post_save=(save_step is not None and step_id > save_step),
                )
            )

            if save_step is not None and torch.cuda.is_available():
                if not save_state["captured_step1"] and step_id == save_step + 1:
                    metrics.memory_snapshots.append(MemorySnapshot.capture("post_save_step_1"))
                    save_state["captured_step1"] = True
                if not save_state["captured_step2"] and step_id == save_step + 2:
                    metrics.memory_snapshots.append(MemorySnapshot.capture("post_save_step_2"))
                    metrics.memory_snapshots.append(MemorySnapshot.capture("post_save_settled"))
                    save_state["captured_step2"] = True

        return original_end_step_hook()

    process.timer.start = wrapped_timer_start
    process.timer.stop = wrapped_timer_stop
    process.save = wrapped_save
    process.end_step_hook = wrapped_end_step_hook

    def teardown():
        process.timer.start = original_timer_start
        process.timer.stop = original_timer_stop
        process.save = original_save
        process.end_step_hook = original_end_step_hook

    teardowns.append(teardown)
    return teardowns


def _cleanup_training_outputs(config: dict) -> list[str]:
    """Remove previous training run folders so benchmarks start from step 0."""
    removed: list[str] = []
    cfg = config.get("config", {})
    run_name = cfg.get("name", "perf_benchmark")
    for process in cfg.get("process", []):
        training_folder = process.get("training_folder")
        if not training_folder:
            continue
        target = Path(training_folder) / run_name
        if target.exists():
            shutil.rmtree(target)
            removed.append(str(target))
    return removed


def _find_latest_lora_output(config: dict) -> Optional[Path]:
    cfg = config.get("config", {})
    run_name = cfg.get("name", "perf_benchmark")
    candidates: list[Path] = []
    for process in cfg.get("process", []):
        training_folder = process.get("training_folder")
        if not training_folder:
            continue
        run_dir = Path(training_folder) / run_name
        if run_dir.exists():
            candidates.extend(run_dir.glob("*.safetensors"))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _normalize_lora_pair_key(key: str, marker: str) -> str:
    # e.g. ".lora_A.weight", ".lora_down.weight", ".lora_B"
    pattern = rf"{re.escape(marker)}(\.weight)?$"
    return re.sub(pattern, "", key)


def _validate_lora_output(path: Path) -> tuple[bool, str]:
    try:
        from safetensors import safe_open
    except Exception as exc:
        return False, f"validation import failed: {exc}"

    try:
        with safe_open(str(path), framework="pt", device="cpu") as f:
            keys = list(f.keys())
            if not keys:
                return False, "checkpoint is empty"

            # Basic shape sanity on first N tensors to catch obvious corruption.
            for key in keys[: min(64, len(keys))]:
                shape = tuple(f.get_tensor(key).shape)
                if any(dim <= 0 for dim in shape):
                    return False, f"invalid tensor shape for key '{key}': {shape}"

            a_keys = [k for k in keys if ".lora_A" in k]
            b_keys = [k for k in keys if ".lora_B" in k]
            down_keys = [k for k in keys if ".lora_down" in k]
            up_keys = [k for k in keys if ".lora_up" in k]

            pair_count = 0
            if a_keys or b_keys:
                a_norm = {_normalize_lora_pair_key(k, ".lora_A") for k in a_keys}
                b_norm = {_normalize_lora_pair_key(k, ".lora_B") for k in b_keys}
                pair_count += len(a_norm & b_norm)
            if down_keys or up_keys:
                d_norm = {_normalize_lora_pair_key(k, ".lora_down") for k in down_keys}
                u_norm = {_normalize_lora_pair_key(k, ".lora_up") for k in up_keys}
                pair_count += len(d_norm & u_norm)

            if pair_count <= 0:
                return False, "no valid LoRA key pairs found"

            return True, f"valid LoRA safetensors ({len(keys)} tensors, {pair_count} paired modules)"
    except Exception as exc:
        return False, f"failed to load safetensors: {exc}"


def _build_generation_validation_config(training_config: dict, lora_path: Path) -> dict:
    cfg = training_config.get("config", {})
    processes = cfg.get("process", [])
    if not processes:
        raise ValueError("training config has no process entries")
    proc = processes[0]

    model_cfg = dict(proc.get("model", {}))
    model_cfg["lora_path"] = str(lora_path)

    resolution = 256
    datasets = proc.get("datasets", [])
    if datasets and datasets[0].get("resolution"):
        res = datasets[0]["resolution"][0]
        if isinstance(res, int):
            resolution = res

    sample_cfg = proc.get("sample", {})
    sampler = sample_cfg.get("sampler", "flowmatch")

    output_folder = proc.get("training_folder", "output/perf_test")
    output_folder = str(Path(output_folder) / "_benchmark_generation_validation")

    validation_prompt = "a portrait photo of a person in natural light"

    return {
        "job": "generate",
        "config": {
            "name": "perf_generation_validation",
            "device": proc.get("device", "cuda:0"),
            "process": [
                {
                    "type": "to_folder",
                    "device": proc.get("device", "cuda:0"),
                    "output_folder": output_folder,
                    "dtype": "bf16",
                    "model": model_cfg,
                    "generate": {
                        "sampler": sampler,
                        "prompts": [validation_prompt],
                        "width": resolution,
                        "height": resolution,
                        "sample_steps": 8,
                        "guidance_scale": 3.5,
                        "seed": 42,
                        "ext": "png",
                    },
                }
            ],
        },
        "meta": {
            "name": "perf_generation_validation",
            "version": "1.0",
        },
    }


def _validate_generated_image(path: Path) -> tuple[bool, str]:
    try:
        if not path.exists() or path.stat().st_size == 0:
            return False, "generated image missing or empty"
        with Image.open(path) as img:
            arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
        std = float(arr.std())
        mean = float(arr.mean())
        if not np.isfinite(std) or not np.isfinite(mean):
            return False, "image stats are non-finite"
        if std < 0.02:
            return False, f"image variance too low (std={std:.4f})"
        if arr.min() == arr.max():
            return False, "image is constant"
        return True, f"image looks non-degenerate (mean={mean:.3f}, std={std:.3f})"
    except Exception as exc:
        return False, f"failed to validate generated image: {exc}"


def _compute_clipscore(image_path: Path, prompt: str, model_id: str) -> tuple[bool, float, str]:
    model = None
    processor = None
    try:
        from transformers import CLIPModel, CLIPProcessor
    except Exception as exc:
        return False, 0.0, f"CLIP dependencies unavailable: {exc}"

    try:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = CLIPModel.from_pretrained(model_id).to(device)
        processor = CLIPProcessor.from_pretrained(model_id)
        with Image.open(image_path).convert("RGB") as img:
            inputs = processor(text=[prompt], images=[img], return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
            text = outputs.text_embeds
            image = outputs.image_embeds
            text = text / text.norm(dim=-1, keepdim=True)
            image = image / image.norm(dim=-1, keepdim=True)
            score = float((image * text).sum(dim=-1).item())
        return True, score, ""
    except Exception as exc:
        return False, 0.0, f"failed to compute CLIPScore: {exc}"
    finally:
        del model
        del processor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _run_generation_smoke_validation(
    training_config: dict,
    lora_path: Path,
    clipscore_threshold: Optional[float],
    clipscore_model: str,
) -> tuple[bool, str, str, Optional[float]]:
    from toolkit.job import get_job

    gen_cfg = _build_generation_validation_config(training_config, lora_path)
    output_folder = Path(
        gen_cfg["config"]["process"][0]["output_folder"]
    )
    output_folder.mkdir(parents=True, exist_ok=True)

    for stale in output_folder.glob("*"):
        if stale.is_file():
            stale.unlink()

    job = get_job(gen_cfg)
    try:
        job.run()
    finally:
        job.cleanup()

    images = sorted(
        [p for p in output_folder.glob("*.png")] + [p for p in output_folder.glob("*.jpg")],
        key=lambda p: p.stat().st_mtime,
    )
    if not images:
        return False, "generation produced no image files", "", None
    latest = images[-1]
    ok, msg = _validate_generated_image(latest)
    if not ok:
        return ok, msg, str(latest), None

    clipscore: Optional[float] = None
    if clipscore_threshold is not None:
        prompt = gen_cfg["config"]["process"][0]["generate"]["prompts"][0]
        clip_ok, score, clip_err = _compute_clipscore(latest, prompt, clipscore_model)
        if not clip_ok:
            return False, clip_err, str(latest), None
        clipscore = score
        if score < clipscore_threshold:
            return (
                False,
                f"CLIPScore below threshold ({score:.4f} < {clipscore_threshold:.4f})",
                str(latest),
                clipscore,
            )
        msg = f"{msg}; CLIPScore={score:.4f} (threshold={clipscore_threshold:.4f})"

    return True, msg, str(latest), clipscore


def get_git_commit() -> str:
    """Get current git commit hash."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
            cwd=PROJECT_ROOT
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return "unknown"


def run_memory_only_benchmark() -> BenchmarkResult:
    """
    Test memory allocation patterns without actual training.
    Useful for testing memory management code in isolation.
    """
    print("Running memory-only benchmark...")
    result = BenchmarkResult(
        tag="memory-only",
        commit=get_git_commit(),
        timestamp=datetime.now().isoformat(),
        hardware=HardwareInfo.capture(),
        config={"mode": "memory-only"},
    )

    if not torch.cuda.is_available():
        print("CUDA not available, skipping memory test")
        return result

    # Import memory management modules
    try:
        from jobs.process.BaseSDTrainProcess import flush
    except ImportError as e:
        print(f"Could not import memory modules: {e}")
        return result

    device = torch.device("cuda:0")

    # Capture baseline
    torch.cuda.empty_cache()
    gc.collect()
    result.metrics.memory_snapshots.append(MemorySnapshot.capture("baseline"))

    # Simulate allocations (like model loading)
    print("  Simulating model allocation...")
    tensors = []
    for i in range(10):
        # Allocate ~100MB chunks
        t = torch.randn(25_000_000, device=device, dtype=torch.float32)
        tensors.append(t)
    result.metrics.memory_snapshots.append(MemorySnapshot.capture("after_alloc"))

    # Simulate training step memory patterns
    print("  Simulating training memory patterns...")
    for step in range(5):
        # Forward pass allocation
        grad_tensors = [torch.randn_like(t) for t in tensors[:3]]
        result.metrics.memory_snapshots.append(MemorySnapshot.capture(f"step_{step}_forward"))

        # Backward pass
        del grad_tensors
        torch.cuda.empty_cache()

    # Simulate save
    print("  Simulating save cycle...")
    result.metrics.memory_snapshots.append(MemorySnapshot.capture("pre_save"))

    # Mimic what save() does
    flush(clear_bouncing_buffers=True, synchronize=True)

    # Clone some state (like state_dict operations)
    state_copies = [t.clone().cpu() for t in tensors[:3]]
    result.metrics.memory_snapshots.append(MemorySnapshot.capture("during_save"))

    del state_copies
    flush(clear_bouncing_buffers=True, synchronize=True)
    result.metrics.memory_snapshots.append(MemorySnapshot.capture("post_save"))

    # Wait and measure settled state
    time.sleep(0.5)
    gc.collect()
    torch.cuda.empty_cache()
    result.metrics.memory_snapshots.append(MemorySnapshot.capture("post_save_settled"))

    # Cleanup
    del tensors
    flush(clear_bouncing_buffers=True, synchronize=True)
    result.metrics.memory_snapshots.append(MemorySnapshot.capture("final_cleanup"))

    result.metrics.compute_summary()
    return result


def run_training_benchmark(
    config_path: Optional[str] = None,
    steps: int = 20,
    resolution: int = 256,
    save_at: Optional[int] = None,
    dataset_path: Optional[str] = None,
    clean_output: bool = False,
    validate_output: bool = True,
    validate_generation: bool = False,
    clipscore_threshold: Optional[float] = None,
    clipscore_model: str = "openai/clip-vit-base-patch32",
) -> BenchmarkResult:
    """
    Run actual training benchmark with timing and memory measurements.

    Args:
        config_path: Path to YAML config, or None to use defaults
        steps: Number of training steps
        resolution: Image resolution
        save_at: Step number to trigger checkpoint save (None = no save)
        dataset_path: Path to dataset folder
    """
    print(f"Running training benchmark: {steps} steps @ {resolution}px")
    if save_at:
        print(f"  Save checkpoint at step {save_at}")

    result = BenchmarkResult(
        tag="training",
        commit=get_git_commit(),
        timestamp=datetime.now().isoformat(),
        hardware=HardwareInfo.capture(),
        config={
            "config_path": config_path,
            "steps": steps,
            "resolution": resolution,
            "save_at": save_at,
            "dataset_path": dataset_path,
        },
    )

    # For now, this is a stub that would integrate with the actual training loop
    # Full implementation requires hooking into BaseSDTrainProcess

    if config_path is None:
        # Use default minimal config
        config_path = str(PROJECT_ROOT / "config" / "perf_test.yaml")

    if not Path(config_path).exists():
        print(f"Config not found: {config_path}")
        print("Create config/perf_test.yaml or specify --config")
        print("Running memory-only benchmark instead...")
        return run_memory_only_benchmark()

    # Import and run training
    try:
        from toolkit.config import get_config
        from toolkit.job import get_job

        # Load and modify config
        config = get_config(config_path, None)

        # Override steps and save settings
        if "process" in config.get("config", {}):
            for proc in config["config"]["process"]:
                if "train" in proc:
                    proc["train"]["steps"] = steps
                    proc["train"]["skip_first_sample"] = True
                    proc["train"]["disable_sampling"] = True
                if "save" in proc:
                    if save_at:
                        proc["save"]["save_every"] = save_at
                    else:
                        proc["save"]["save_every"] = steps + 1  # Never save
                if "sample" in proc:
                    proc["sample"]["sample_every"] = steps + 1  # Never sample
                # Override resolution
                if "datasets" in proc:
                    for ds in proc["datasets"]:
                        ds["resolution"] = [resolution]

        if clean_output:
            removed_paths = _cleanup_training_outputs(config)
            if removed_paths:
                print("  Cleaned previous benchmark outputs:")
                for path in removed_paths:
                    print(f"    - {path}")
            else:
                print("  No previous benchmark outputs to clean")

        # Capture pre-training memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
            torch.cuda.reset_peak_memory_stats()
        result.metrics.memory_snapshots.append(MemorySnapshot.capture("pre_training"))

        start_time = time.time()

        # Create and run job
        job = get_job(config)
        job_cleaned = False
        teardowns = _instrument_training_processes(job, result.metrics, save_at)

        print("  Starting training...")
        try:
            job.run()
        finally:
            for teardown in teardowns:
                teardown()

        result.metrics.total_duration_s = time.time() - start_time

        # Capture post-training memory
        result.metrics.memory_snapshots.append(MemorySnapshot.capture("post_training"))

        if validate_output:
            latest_output = _find_latest_lora_output(config)
            if latest_output is None:
                result.metrics.output_validation_passed = False
                result.metrics.output_validation_message = "no .safetensors output found to validate"
            else:
                ok, message = _validate_lora_output(latest_output)
                result.metrics.output_validation_passed = ok
                result.metrics.output_validation_message = message
                result.metrics.validated_output_path = str(latest_output)
                status = "PASS" if ok else "FAIL"
                print(f"  Output validation [{status}]: {message}")
                print(f"  Validated file: {latest_output}")

                if validate_generation:
                    # Release training resources before generation validation to avoid
                    # carrying training VRAM footprint into generation.
                    job.cleanup()
                    job_cleaned = True
                    del job
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.ipc_collect()

                    ok, msg, image_path, clipscore = _run_generation_smoke_validation(
                        config,
                        latest_output,
                        clipscore_threshold=clipscore_threshold,
                        clipscore_model=clipscore_model,
                    )
                    result.metrics.generation_validation_passed = ok
                    result.metrics.generation_validation_message = msg
                    result.metrics.generation_validation_image_path = image_path
                    if clipscore is not None:
                        result.metrics.generation_clipscore = clipscore
                        result.metrics.generation_clipscore_threshold = clipscore_threshold or 0.0
                        result.metrics.generation_clipscore_model = clipscore_model
                    gstatus = "PASS" if ok else "FAIL"
                    print(f"  Generation validation [{gstatus}]: {msg}")
                    if image_path:
                        print(f"  Generated file: {image_path}")

        # Fallback estimate if hooks did not produce step-level timings
        if steps > 0 and not result.metrics.step_times:
            result.metrics.avg_step_time_s = result.metrics.total_duration_s / steps

        if not job_cleaned:
            job.cleanup()

    except Exception as e:
        print(f"Training benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        return result

    result.metrics.compute_summary(save_at)
    return result


def print_result_summary(result: BenchmarkResult):
    """Print human-readable summary of benchmark results."""
    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS")
    print("=" * 60)

    print(f"\nHardware: {result.hardware.gpu}")
    print(f"VRAM: {result.hardware.vram_gb:.1f} GB")
    print(f"Commit: {result.commit}")
    print(f"Timestamp: {result.timestamp}")

    m = result.metrics
    print("\nTiming:")
    print(f"  Total duration:     {m.total_duration_s:.2f}s")
    print(f"  Avg step time:      {m.avg_step_time_s:.3f}s")
    if m.pre_save_step_time_s > 0:
        print(f"  Pre-save step:      {m.pre_save_step_time_s:.3f}s")
    if m.post_save_step_time_s > 0:
        print(f"  Post-save step:     {m.post_save_step_time_s:.3f}s")
        print(f"  Slowdown ratio:     {m.slowdown_ratio:.2f}x")
    if m.save_duration_s > 0:
        print(f"  Save duration:      {m.save_duration_s:.3f}s")

    print("\nMemory:")
    print(f"  Peak VRAM:          {m.peak_vram_mb:.0f} MB")
    if m.pre_save_vram_mb > 0:
        print(f"  Pre-save VRAM:      {m.pre_save_vram_mb:.0f} MB")
    if m.save_peak_vram_mb > 0:
        print(f"  Save peak VRAM:     {m.save_peak_vram_mb:.0f} MB")
    if m.post_save_vram_mb > 0:
        print(f"  Post-save VRAM:     {m.post_save_vram_mb:.0f} MB")
        print(f"  Leaked:             {m.leaked_mb:.0f} MB")
    print(f"  Alloc retries:      {m.num_alloc_retries}")
    print(f"  Fragmentation:      {m.fragmentation_ratio:.1%}")
    if m.output_validation_message:
        status = "PASS" if m.output_validation_passed else "FAIL"
        print("\nOutput validation:")
        print(f"  Status:             {status}")
        print(f"  Message:            {m.output_validation_message}")
        if m.validated_output_path:
            print(f"  File:               {m.validated_output_path}")
    if m.generation_validation_message:
        status = "PASS" if m.generation_validation_passed else "FAIL"
        print("\nGeneration validation:")
        print(f"  Status:             {status}")
        print(f"  Message:            {m.generation_validation_message}")
        if m.generation_validation_image_path:
            print(f"  File:               {m.generation_validation_image_path}")
        if m.generation_clipscore_model:
            print(f"  CLIP model:         {m.generation_clipscore_model}")
            print(f"  CLIPScore:          {m.generation_clipscore:.4f}")
            print(f"  Threshold:          {m.generation_clipscore_threshold:.4f}")

    print("\nMemory snapshots:")
    for snap in result.metrics.memory_snapshots:
        print(f"  {snap.label:20s}: {snap.reserved_mb:8.0f} MB reserved, {snap.allocated_mb:8.0f} MB allocated")

    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="AI Toolkit Performance Benchmark")

    parser.add_argument("--config", "-c", type=str, default=None,
                        help="Path to training config YAML")
    parser.add_argument("--steps", "-s", type=int, default=20,
                        help="Number of training steps (default: 20)")
    parser.add_argument("--resolution", "-r", type=int, default=256,
                        help="Image resolution (default: 256)")
    parser.add_argument("--save-at", type=int, default=None,
                        help="Step to trigger checkpoint save")
    parser.add_argument("--dataset", "-d", type=str, default=None,
                        help="Path to dataset folder")
    parser.add_argument("--memory-only", action="store_true",
                        help="Run memory-only test (no actual training)")
    parser.add_argument("--clean-output", action="store_true",
                        help="Remove prior training_folder/name output before running")
    parser.add_argument("--no-validate-output", action="store_true",
                        help="Skip post-run LoRA safetensors validation")
    parser.add_argument("--validate-generation", action="store_true",
                        help="Run post-run generation smoke validation using saved LoRA")
    parser.add_argument("--clipscore-threshold", type=float, default=0.20,
                        help="Minimum CLIPScore when --validate-generation is enabled")
    parser.add_argument("--clipscore-model", type=str, default="openai/clip-vit-base-patch32",
                        help="CLIP model ID for CLIPScore calculation")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Output JSON file path")
    parser.add_argument("--tag", "-t", type=str, default="",
                        help="Tag for this benchmark run")
    parser.add_argument("--quiet", "-q", action="store_true",
                        help="Suppress verbose output")

    args = parser.parse_args()

    # Run benchmark
    if args.memory_only:
        result = run_memory_only_benchmark()
    else:
        result = run_training_benchmark(
            config_path=args.config,
            steps=args.steps,
            resolution=args.resolution,
            save_at=args.save_at,
            dataset_path=args.dataset,
            clean_output=args.clean_output,
            validate_output=not args.no_validate_output,
            validate_generation=args.validate_generation,
            clipscore_threshold=(args.clipscore_threshold if args.validate_generation else None),
            clipscore_model=args.clipscore_model,
        )

    if args.tag:
        result.tag = args.tag

    # Print summary
    if not args.quiet:
        print_result_summary(result)

    # Save to file
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(result.to_dict(), f, indent=2)
        print(f"\nResults saved to: {output_path}")

    # Also print JSON to stdout if no output file specified
    if args.output is None and args.quiet:
        print(json.dumps(result.to_dict(), indent=2))

    return result


if __name__ == "__main__":
    main()

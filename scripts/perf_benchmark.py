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

import torch  # noqa: E402


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

        # Capture pre-training memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
            torch.cuda.reset_peak_memory_stats()
        result.metrics.memory_snapshots.append(MemorySnapshot.capture("pre_training"))

        start_time = time.time()

        # Create and run job
        job = get_job(config)
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

        # Fallback estimate if hooks did not produce step-level timings
        if steps > 0 and not result.metrics.step_times:
            result.metrics.avg_step_time_s = result.metrics.total_duration_s / steps

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

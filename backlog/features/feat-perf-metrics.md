# feat-perf-metrics

## Problem
Current `loss_log.db` captures time-series values (`steps.wall_time`, `metrics`) but does not explicitly label non-training overhead (sampling, checkpoint save, stable-loss pass, OOM retry). This makes run-to-run performance comparisons noisy and hard to trust.

## Goal
Make training runs directly comparable with:
- `train_only_sec_per_step` (excludes sample/save/stable-loss overhead)
- clear run metadata (hardware, git, config fingerprint)
- consistent per-run summary metrics that can be queried from DB

## Non-goals
- Replacing existing `loss_log.db` APIs/UI.
- Building a full benchmark harness (already exists in `scripts/perf_benchmark.py`).

## Existing Signals (today)
- `loss_log.db`
  - `steps(step, wall_time)`
  - `metrics(step, key, value_real, value_text)`
  - `metric_keys(key, first_seen_step, last_seen_step)`
- System metrics already logged in `BaseSDTrainProcess`:
  - `vram_gb`, `ram_gb`, `cpu_percent`
- Some runs log `loss/stable`, but no explicit event markers for sample/save.

## Proposed Design

### 1) Add Event Markers Per Step
At each committed step, also log boolean/int event keys:
- `event/is_update_step` (`0|1`)
- `event/is_grad_accum_step` (`0|1`)
- `event/is_sample_step` (`0|1`)
- `event/is_save_step` (`0|1`)
- `event/is_stable_loss_step` (`0|1`)
- `event/is_oom_recovery_step` (`0|1`)
- `event/was_skipped_step` (`0|1`)

And useful context keys:
- `perf/step_wall_s_raw` (current step duration)
- `perf/lr_scheduler_step_wall_s` (optional)
- `perf/optimizer_step_wall_s` (optional)
- `perf/backward_wall_s` (optional)

Implementation point: `jobs/process/BaseSDTrainProcess.py` inside the main step loop, before `self.logger.commit(step=self.step_num)`.

### 2) Add Run Metadata Table (loss_log.db)
Extend `UILogger` schema with:

```sql
CREATE TABLE IF NOT EXISTS run_info (
  key TEXT PRIMARY KEY,
  value_text TEXT,
  value_real REAL
);
```

Populate once at run start (`logger.start()` or first commit):
- `run_id`, `job_id`, `job_name`
- `started_at_unix`, `ended_at_unix`
- `git_commit`, `git_branch`, `git_dirty`
- `gpu_name`, `gpu_vram_gb`, `torch_version`, `cuda_version`
- `train_steps_target`, `batch_size`, `gradient_accumulation`, `dtype`
- `model_name_or_path`, `model_arch`
- `config_sha256` (canonicalized config hash)

This keeps each run self-describing, even if `aitk_db.db` fields are inconsistent.

### 3) Add Run Summary Table (loss_log.db)
Compute once at run end and persist:

```sql
CREATE TABLE IF NOT EXISTS run_summary (
  key TEXT PRIMARY KEY,
  value_real REAL,
  value_text TEXT
);
```

Required summary fields:
- `summary/steps_logged`
- `summary/steps_completed`
- `summary/train_only_steps`
- `summary/elapsed_wall_s_total`
- `summary/sec_per_step_raw`
- `summary/sec_per_step_train_only`
- `summary/sec_per_update_step`
- `summary/vram_gb_p50`, `summary/vram_gb_p95`, `summary/vram_gb_peak`
- `summary/cpu_percent_p50`, `summary/ram_gb_p95`
- `summary/sample_events`, `summary/save_events`, `summary/stable_loss_events`

## Train-only Metric Definition
`train_only_sec_per_step` should be computed from adjacent step deltas where all are true:
- `event/is_update_step = 1` (or configurable, if you want micro-step throughput too)
- `event/is_sample_step = 0`
- `event/is_save_step = 0`
- `event/is_stable_loss_step = 0`
- `event/is_oom_recovery_step = 0`
- `event/was_skipped_step = 0`

Also exclude obvious pauses with a robust cap:
- `delta_s <= p95(delta_s_non_event) + 3 * IQR`

Store final value in `run_summary`.

## Comparison Contract
Two runs are comparable only if these fields match (or are explicitly waived):
- `model_arch`
- `dtype`
- `batch_size`
- `gradient_accumulation`
- `resolution bucket policy` (if available)
- dataset fingerprint (`dataset_path` + optional file manifest hash)

If mismatched, comparison should be marked `non_comparable` in output.

## Minimal Query Examples

### Latest comparable runs with train-only throughput
```sql
SELECT
  ri1.value_text AS run_name,
  rs.value_real AS train_only_sec_per_step
FROM run_summary rs
JOIN run_info ri1 ON ri1.key = 'job_name'
WHERE rs.key = 'summary/sec_per_step_train_only'
ORDER BY (SELECT value_real FROM run_info WHERE key='started_at_unix') DESC
LIMIT 10;
```

### Event overhead counts per run
```sql
SELECT
  MAX(CASE WHEN key='summary/sample_events' THEN value_real END) AS sample_events,
  MAX(CASE WHEN key='summary/save_events' THEN value_real END) AS save_events,
  MAX(CASE WHEN key='summary/stable_loss_events' THEN value_real END) AS stable_events
FROM run_summary;
```

## Implementation Plan
1. `UILogger` schema extension (`run_info`, `run_summary`), backward compatible.
2. Instrument event keys in `BaseSDTrainProcess` per step.
3. Write start/end metadata + compute summary on `logger.finish()`.
4. Add a small utility script: `scripts/summarize_run_metrics.py` to print comparable metrics from a run folder.
5. Update UI/API (optional second pass) to display `train_only_sec_per_step` and event counts.

## Acceptance Criteria
- For the last 5 runs, one command produces:
  - raw sec/step
  - train-only sec/step
  - sample/save/stable-loss counts
  - comparable/non-comparable flag
- No manual filtering needed.
- Existing `/api/jobs/[jobID]/loss` endpoints continue to work.

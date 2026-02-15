# Feature: Job Elapsed Time Tracking

**Status:** Proposed
**Priority:** Low
**Complexity:** Low-Medium
**Category:** UI / Training Observability

---

## Summary

Track and display cumulative wall-clock time a job has actually spent training, correctly accounting for pause/resume cycles. Currently the UI shows ETA (estimated time remaining) but not how long a job has been running.

---

## Problem Statement

When a training job is running, users have no way to see how long it has been actively training. The UI shows:
- ETA / remaining time (estimated)
- Current step vs. total steps
- Speed (iter/sec or sec/iter)

What it **doesn't** show:
- How long the job has been running in total
- How long the current run session has been active

**Why "now minus created_at" doesn't work:**

Jobs can be paused (stopped) and resumed. A job created Monday, paused overnight, and resumed Tuesday hasn't been training for 24+ hours. The system needs to accumulate only the time the job was actually in `running` status.

**Current status model:** `stopped -> queued -> running -> stopped` (stop = pause, start = resume from checkpoint). There is no dedicated "paused" state.

---

## Proposed Solution

### Approach: DB-Level Accumulation

Add two fields to the Job model that track cumulative training time across pause/resume cycles:

| Field | Type | Purpose |
|-------|------|---------|
| `running_since` | DateTime? | Timestamp when current run session started (null if not running) |
| `total_running_seconds` | Int | Accumulated seconds from all previous run sessions |

**Display formula:** If currently running: `total_running_seconds + (now - running_since)`. If stopped: `total_running_seconds`.

### Why Not Derive from Loss Data?

The loss database records `wall_time` per step, and you could theoretically sum inter-step deltas while filtering out pause gaps. However:
- Only works once steps start logging (not during model/dataset loading)
- Requires heuristic gap detection (what threshold = "paused"?)
- More fragile and approximate than explicit tracking

---

## Technical Architecture

### Backend Changes

#### 1. Prisma Schema: `ui/prisma/schema.prisma`

```prisma
model Job {
  // ... existing fields ...
  running_since          DateTime?
  total_running_seconds  Int       @default(0)
}
```

#### 2. Status Transition Points

Update the following locations to set/accumulate timing:

**On transition to `running`:**
- `ui/cron/actions/startJob.ts` — When spawning the training process
- Set `running_since = now()`

```typescript
data: {
  status: 'running',
  stop: false,
  info: 'Starting job...',
  running_since: new Date(),
}
```

**On transition away from `running` (stopped, completed, error):**
- `extensions_built_in/sd_trainer/UITrainer.py` — `maybe_stop()`, `done_hook()`, `on_error()`
- Accumulate elapsed time into `total_running_seconds`, clear `running_since`

```python
def _accumulate_running_time(self):
    """Add current session's elapsed time to total and clear running_since."""
    with self._db_connect() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT running_since FROM Job WHERE id = ?",
            (self.job_id,)
        )
        row = cursor.fetchone()
        if row and row[0]:
            started = datetime.fromisoformat(row[0])
            elapsed = int((datetime.now(timezone.utc) - started).total_seconds())
            cursor.execute(
                "UPDATE Job SET total_running_seconds = total_running_seconds + ?, running_since = NULL WHERE id = ?",
                (elapsed, self.job_id)
            )
            conn.commit()
```

Call this in `maybe_stop()`, `done_hook()`, and `on_error()`.

**Edge case — process crash without clean exit:**
- `startJob.ts` subprocess `close` handler already catches abnormal exits
- Add accumulation there as a safety net

```typescript
subprocess.on('close', async (code) => {
  const job = await prisma.job.findUnique({ where: { id: jobID } });
  if (job?.running_since) {
    const elapsed = Math.floor((Date.now() - job.running_since.getTime()) / 1000);
    await prisma.job.update({
      where: { id: jobID },
      data: {
        total_running_seconds: { increment: elapsed },
        running_since: null,
      },
    });
  }
  // ... existing error handling ...
});
```

#### 3. API: Job Response

No new endpoint needed. The existing job polling already returns all Job fields. The frontend computes display value from `total_running_seconds` and `running_since`.

### Frontend Changes

#### 1. Display Utility: `ui/src/utils/formatElapsedTime.ts`

```typescript
export function getJobElapsedSeconds(job: Job): number {
  let total = job.total_running_seconds ?? 0;
  if (job.running_since && job.status === 'running') {
    const since = new Date(job.running_since).getTime();
    total += Math.floor((Date.now() - since) / 1000);
  }
  return total;
}

export function formatElapsedTime(seconds: number): string {
  if (seconds < 60) return `${seconds}s`;
  const hours = Math.floor(seconds / 3600);
  const minutes = Math.floor((seconds % 3600) / 60);
  if (hours === 0) return `${minutes}m`;
  return `${hours}h ${minutes}m`;
}
```

#### 2. Display Location: `ui/src/components/JobOverview.tsx`

Add elapsed time to the job overview panel, near the existing speed and progress info:

```
Status: Running          Speed: 1.23 iter/sec
Step: 1500 / 3000       Elapsed: 2h 34m
ETA: 1h 17m
```

Ticks every second while the job is running (use `setInterval` or existing polling).

#### 3. Display Location: `ui/src/components/JobsTableExtended.tsx`

Add an "Elapsed" column to the jobs table for at-a-glance comparison across jobs.

---

## Migration

Existing jobs will have `total_running_seconds = 0` and `running_since = null`. This is accurate for stopped/completed jobs (we simply don't have historical data) and acceptable for currently-running jobs (they'll start accumulating from the migration point forward).

---

## Implementation Phases

### Phase 1: Core Tracking (MVP)
- [ ] Add `running_since` and `total_running_seconds` to Prisma schema
- [ ] Run migration
- [ ] Set `running_since` in `startJob.ts` on transition to running
- [ ] Accumulate time in UITrainer on stop/complete/error
- [ ] Accumulate time in subprocess crash handler

### Phase 2: UI Display
- [ ] Add `formatElapsedTime` utility
- [ ] Display elapsed time in JobOverview
- [ ] Add elapsed column to JobsTableExtended
- [ ] Live-tick the counter while job is running

---

## Performance Considerations

- No measurable overhead — just one extra timestamp write on start and one read+write on stop
- Frontend ticking is purely client-side math, no additional API calls
- No impact on training loop performance

---

## Open Questions

1. **Reset on re-run?** If a user stops a job at step 1000, changes the config, and starts it again from step 0, should elapsed time reset? Probably yes — but detecting "fresh start vs. resume" may need a heuristic (e.g., reset if step goes backward).

2. **Per-session breakdown?** Worth showing a list of run sessions (start/stop times) in the job detail view? Adds complexity but could be useful for debugging.

3. **Include queue time?** Should there be a separate "time in queue" display? Low priority but easy to add with the same pattern.

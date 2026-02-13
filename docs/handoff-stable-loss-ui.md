# Handoff: Stable Loss UI Implementation

## Summary

The stable loss feature (deterministic validation loss tracking) is implemented in the backend but not exposed in the UI job creation form. This document provides everything needed to complete the UI integration.

## What's Already Done

| Component | File | Status |
|-----------|------|--------|
| Stable loss computation | `extensions_built_in/sd_trainer/SDTrainer.py` | ✅ Complete |
| Config options | `toolkit/config_modules.py` | ✅ Complete |
| Loss graph display | `ui/src/components/JobLossGraph.tsx` | ✅ Complete |
| **UI job creation form** | `ui/src/app/jobs/new/SimpleJob.tsx` | ❌ **TODO** |
| TypeScript types | `ui/src/types.ts` | ❌ **TODO** |
| Default config values | `ui/src/app/jobs/new/jobConfig.ts` | ❌ **TODO** |
| Documentation tooltips | `ui/src/docs.tsx` | ❌ **TODO** |

## Backend Config Options (already implemented)

From `toolkit/config_modules.py`:

```python
self.stable_loss_enabled: bool = kwargs.get('stable_loss_enabled', False)
self.stable_loss_path: Optional[str] = kwargs.get('stable_loss_path', None)  # path to 1-2 representative images
self.stable_loss_steps: int = kwargs.get('stable_loss_steps', 100)  # evaluate every N steps
self.stable_loss_seed: int = kwargs.get('stable_loss_seed', 1234)  # fixed seed for deterministic eval
self.stable_loss_repeats: int = kwargs.get('stable_loss_repeats', 4)  # timestep buckets for uniform coverage
```

## Files to Modify

### 1. `ui/src/types.ts`

Add to the `Train` interface (around line 147):

```typescript
// Stable loss (deterministic validation)
stable_loss_enabled?: boolean;
stable_loss_path?: string;
stable_loss_steps?: number;
stable_loss_seed?: number;
stable_loss_repeats?: number;
```

### 2. `ui/src/app/jobs/new/jobConfig.ts`

Add default values in the `train` section (around line 92):

```typescript
// Stable loss defaults
stable_loss_enabled: false,
stable_loss_steps: 100,
stable_loss_seed: 1234,
stable_loss_repeats: 4,
```

### 3. `ui/src/docs.tsx`

Add documentation entry:

```tsx
'train.stable_loss': {
  title: 'Stable Loss (Validation)',
  description: (
    <>
      Stable Loss computes deterministic validation loss on held-out images to track true training progress.
      Unlike noisy per-step training loss, stable loss produces clean curves that show when the model is
      actually learning vs overfitting.
      <br />
      <br />
      <strong>How it works:</strong> At regular intervals, the model predicts noise on your validation images
      using fixed random seeds and uniform timestep coverage. This produces reproducible loss values that
      reveal the U-shaped curve: loss drops as the model learns, then rises as it begins overfitting.
      <br />
      <br />
      <strong>The minimum of the stable loss curve is your optimal stopping point.</strong>
      <br />
      <br />
      Provide 1-3 representative images from your dataset (or similar images you hold out from training).
      More images = smoother signal, but 1-2 is usually sufficient.
    </>
  ),
},
```

### 4. `ui/src/app/jobs/new/SimpleJob.tsx`

Add UI controls in the "Advanced" card section (around line 750, after Differential Guidance).

Follow the pattern used by Differential Output Preservation:

```tsx
{/* Stable Loss Section */}
<div>
  <Checkbox
    label="Enable Stable Loss"
    docKey={'train.stable_loss'}
    className="pt-1"
    checked={jobConfig.config.process[0].train.stable_loss_enabled || false}
    onChange={value => {
      setJobConfig(value, 'config.process[0].train.stable_loss_enabled');
      if (!value) {
        setJobConfig(undefined, 'config.process[0].train.stable_loss_path');
      }
    }}
  />
  {jobConfig.config.process[0].train.stable_loss_enabled && (
    <>
      <TextInput
        label="Validation Images Path"
        className="pt-2"
        value={jobConfig.config.process[0].train.stable_loss_path as string || ''}
        onChange={value => setJobConfig(value, 'config.process[0].train.stable_loss_path')}
        placeholder="/path/to/validation/images"
      />
      <NumberInput
        label="Evaluate Every N Steps"
        className="pt-2"
        value={(jobConfig.config.process[0].train.stable_loss_steps as number) || 100}
        onChange={value => setJobConfig(value, 'config.process[0].train.stable_loss_steps')}
        placeholder="100"
        min={10}
      />
      <NumberInput
        label="Seed"
        className="pt-2"
        value={(jobConfig.config.process[0].train.stable_loss_seed as number) || 1234}
        onChange={value => setJobConfig(value, 'config.process[0].train.stable_loss_seed')}
        placeholder="1234"
        min={0}
      />
      <NumberInput
        label="Timestep Buckets"
        className="pt-2"
        value={(jobConfig.config.process[0].train.stable_loss_repeats as number) || 4}
        onChange={value => setJobConfig(value, 'config.process[0].train.stable_loss_repeats')}
        placeholder="4"
        min={1}
        max={10}
      />
    </>
  )}
</div>
```

## UI Location Recommendation

Add the Stable Loss controls in the **Advanced** card section, which already contains:
- Differential Guidance
- (This is where experimental/advanced training features live)

Alternatively, create a new "Validation" subsection within the Training card near EMA settings.

## Testing

1. Start the UI: `cd ui && npm run dev`
2. Create a new job
3. Enable "Stable Loss" checkbox
4. Verify:
   - Path field appears when enabled
   - All numeric fields have sensible defaults
   - Config YAML includes the stable_loss_* fields when saved
5. Run a short training job with stable loss enabled
6. Verify `loss_log.db` contains `stable` metric key
7. Verify JobLossGraph shows "Stable Loss" line in emerald green

## Optional Enhancements

### File/Folder Picker
Instead of a text input for the path, consider adding a file browser component. The datasets already have folder selection - could reuse that pattern.

### Validation Image Preview
Show thumbnails of selected validation images so users can confirm they picked the right ones.

### Auto-suggest from Dataset
Offer to randomly select 2-3 images from the training dataset as validation holdouts (with a warning that they should ideally be removed from training).

## Related Files for Reference

- `ui/src/app/jobs/new/SimpleJob.tsx` - Main job creation form
- `ui/src/components/JobLossGraph.tsx` - Already handles `stable` metric display
- `extensions_built_in/sd_trainer/SDTrainer.py` - `compute_stable_loss()` implementation
- `docs/flux2_klein.md` - Documentation on why stable loss matters

## Contact

Feature implemented on branch: `feat/system-metrics-graph`

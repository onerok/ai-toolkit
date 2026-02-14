'use client';

import useJobMetrics, { JobMetricPoint } from './useJobMetrics';

export type LossPoint = JobMetricPoint;
const LOSS_FALLBACK_KEYS = ['loss'];

function isLossKey(key: string) {
  // treat anything containing "loss" as a loss-series
  // (covers loss, train_loss, val_loss, loss/xyz, etc.)
  return /loss/i.test(key);
}

export default function useJobLossLog(jobID: string, reloadInterval: null | number = null) {
  const { series, setSeries, keys, metricKeys: lossKeys, status, refreshMetrics: refreshLoss } = useJobMetrics({
    jobID,
    reloadInterval,
    keyProbe: 'loss',
    includeKey: isLossKey,
    fallbackKeys: LOSS_FALLBACK_KEYS,
    errorLabel: 'loss logs',
  });

  return { series, keys, lossKeys, status, refreshLoss, setSeries };
}

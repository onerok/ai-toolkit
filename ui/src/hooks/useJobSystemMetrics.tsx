'use client';

import useJobMetrics, { JobMetricPoint } from './useJobMetrics';

export type MetricPoint = JobMetricPoint;

// System metric keys we care about
const SYSTEM_METRIC_KEYS = ['vram_gb', 'ram_gb', 'cpu_percent'];
const SYSTEM_FALLBACK_KEYS: string[] = [];

function isSystemMetricKey(key: string) {
  return SYSTEM_METRIC_KEYS.includes(key);
}

export default function useJobSystemMetrics(jobID: string, reloadInterval: null | number = null) {
  const { series, keys, metricKeys, status, refreshMetrics } = useJobMetrics({
    jobID,
    reloadInterval,
    keyProbe: 'vram_gb',
    includeKey: isSystemMetricKey,
    fallbackKeys: SYSTEM_FALLBACK_KEYS,
    errorLabel: 'system metrics',
  });
  const hasMetrics = metricKeys.length > 0;

  return { series, keys, metricKeys, hasMetrics, status, refreshMetrics };
}

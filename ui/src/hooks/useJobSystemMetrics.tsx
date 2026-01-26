'use client';

import { useEffect, useState, useRef, useCallback, useMemo } from 'react';
import { apiClient } from '@/utils/api';

export interface MetricPoint {
  step: number;
  wall_time?: number;
  value: number | null;
}

type SeriesMap = Record<string, MetricPoint[]>;

// System metric keys we care about
const SYSTEM_METRIC_KEYS = ['vram_gb', 'ram_gb', 'cpu_percent'];

function isSystemMetricKey(key: string) {
  return SYSTEM_METRIC_KEYS.includes(key);
}

export default function useJobSystemMetrics(jobID: string, reloadInterval: null | number = null) {
  const [series, setSeries] = useState<SeriesMap>({});
  const [keys, setKeys] = useState<string[]>([]);
  const [status, setStatus] = useState<'idle' | 'loading' | 'success' | 'error' | 'refreshing'>('idle');

  const didInitialLoadRef = useRef(false);
  const inFlightRef = useRef(false);

  // track last step per key so polling is incremental per series
  const lastStepByKeyRef = useRef<Record<string, number | null>>({});

  const metricKeys = useMemo(() => {
    const base = (keys ?? []).filter(isSystemMetricKey);
    return base.sort();
  }, [keys]);

  const hasMetrics = metricKeys.length > 0;

  const refreshMetrics = useCallback(async () => {
    if (!jobID) return;

    if (inFlightRef.current) return;
    inFlightRef.current = true;

    const loadStatus: 'loading' | 'refreshing' = didInitialLoadRef.current ? 'refreshing' : 'loading';
    setStatus(loadStatus);

    try {
      // Step 1: get key list
      const first = await apiClient
        .get(`/api/jobs/${jobID}/loss`, { params: { key: 'vram_gb', limit: 1 } })
        .then(res => res.data as { keys?: string[] });

      const newKeys = first.keys ?? [];
      setKeys(newKeys);

      const wantedKeys = newKeys.filter(isSystemMetricKey).sort();

      if (wantedKeys.length === 0) {
        // No system metrics available yet
        setStatus('success');
        didInitialLoadRef.current = true;
        inFlightRef.current = false;
        return;
      }

      // Step 2: fetch each key incrementally
      const requests = wantedKeys.map(k => {
        const params: Record<string, any> = { key: k };

        if (reloadInterval && lastStepByKeyRef.current[k] != null) {
          params.since_step = lastStepByKeyRef.current[k];
        }

        return apiClient
          .get(`/api/jobs/${jobID}/loss`, { params })
          .then(res => res.data as { key: string; points?: MetricPoint[] });
      });

      const results = await Promise.all(requests);

      setSeries(prev => {
        const next: SeriesMap = { ...prev };

        for (const r of results) {
          const k = r.key;
          const newPoints = (r.points ?? []).filter(p => p.value !== null);

          if (!didInitialLoadRef.current) {
            next[k] = newPoints;
          } else if (newPoints.length) {
            const existing = next[k] ?? [];
            const prevLast = existing.length ? existing[existing.length - 1].step : null;
            const filtered = prevLast == null ? newPoints : newPoints.filter(p => p.step > prevLast);
            next[k] = filtered.length ? [...existing, ...filtered] : existing;
          } else {
            next[k] = next[k] ?? [];
          }

          const finalArr = next[k] ?? [];
          lastStepByKeyRef.current[k] = finalArr.length
            ? finalArr[finalArr.length - 1].step
            : (lastStepByKeyRef.current[k] ?? null);
        }

        // remove stale keys
        for (const existingKey of Object.keys(next)) {
          if (isSystemMetricKey(existingKey) && !wantedKeys.includes(existingKey)) {
            delete next[existingKey];
            delete lastStepByKeyRef.current[existingKey];
          }
        }

        return next;
      });

      setStatus('success');
      didInitialLoadRef.current = true;
    } catch (err) {
      console.error('Error fetching system metrics:', err);
      setStatus('error');
    } finally {
      inFlightRef.current = false;
    }
  }, [jobID, reloadInterval]);

  useEffect(() => {
    // reset when job changes
    didInitialLoadRef.current = false;
    lastStepByKeyRef.current = {};
    setSeries({});
    setKeys([]);
    setStatus('idle');

    refreshMetrics();

    if (reloadInterval) {
      const interval = setInterval(() => {
        refreshMetrics();
      }, reloadInterval);

      return () => clearInterval(interval);
    }
  }, [jobID, reloadInterval, refreshMetrics]);

  return { series, keys, metricKeys, hasMetrics, status, refreshMetrics };
}

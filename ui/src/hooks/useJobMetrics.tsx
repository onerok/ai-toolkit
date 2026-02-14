'use client';

import { useEffect, useState, useRef, useCallback, useMemo } from 'react';
import { apiClient } from '@/utils/api';

export interface JobMetricPoint {
  step: number;
  wall_time?: number;
  value: number | null;
}

export type JobMetricSeriesMap = Record<string, JobMetricPoint[]>;
export type JobMetricStatus = 'idle' | 'loading' | 'success' | 'error' | 'refreshing';

interface UseJobMetricsOptions {
  jobID: string;
  reloadInterval?: null | number;
  keyProbe: string;
  includeKey: (key: string) => boolean;
  fallbackKeys?: string[];
  errorLabel: string;
}

export default function useJobMetrics({
  jobID,
  reloadInterval = null,
  keyProbe,
  includeKey,
  fallbackKeys = [],
  errorLabel,
}: UseJobMetricsOptions) {
  const [series, setSeries] = useState<JobMetricSeriesMap>({});
  const [keys, setKeys] = useState<string[]>([]);
  const [status, setStatus] = useState<JobMetricStatus>('idle');

  const didInitialLoadRef = useRef(false);
  const inFlightRef = useRef(false);
  const lastStepByKeyRef = useRef<Record<string, number | null>>({});

  const metricKeys = useMemo(() => {
    const filtered = (keys ?? []).filter(includeKey).sort();
    if (filtered.length > 0) return filtered;
    return [...fallbackKeys].sort();
  }, [keys, includeKey, fallbackKeys]);

  const refreshMetrics = useCallback(async () => {
    if (!jobID) return;
    if (inFlightRef.current) return;
    inFlightRef.current = true;

    const loadStatus: 'loading' | 'refreshing' = didInitialLoadRef.current ? 'refreshing' : 'loading';
    setStatus(loadStatus);

    try {
      const first = await apiClient
        .get(`/api/jobs/${jobID}/loss`, { params: { key: keyProbe, limit: 1 } })
        .then(res => res.data as { keys?: string[] });

      const newKeys = first.keys ?? [];
      setKeys(newKeys);

      const filteredKeys = newKeys.filter(includeKey);
      const wantedKeys = (filteredKeys.length > 0 ? filteredKeys : fallbackKeys).sort();

      if (wantedKeys.length === 0) {
        setStatus('success');
        didInitialLoadRef.current = true;
        return;
      }

      const requests = wantedKeys.map(k => {
        const params: Record<string, number | string> = { key: k };
        if (reloadInterval && lastStepByKeyRef.current[k] != null) {
          params.since_step = lastStepByKeyRef.current[k];
        }
        return apiClient
          .get(`/api/jobs/${jobID}/loss`, { params })
          .then(res => res.data as { key: string; points?: JobMetricPoint[] });
      });

      const results = await Promise.all(requests);

      setSeries(prev => {
        const next: JobMetricSeriesMap = { ...prev };

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
          lastStepByKeyRef.current[k] = finalArr.length ? finalArr[finalArr.length - 1].step : null;
        }

        for (const existingKey of Object.keys(next)) {
          if (includeKey(existingKey) && !wantedKeys.includes(existingKey)) {
            delete next[existingKey];
            delete lastStepByKeyRef.current[existingKey];
          }
        }

        return next;
      });

      setStatus('success');
      didInitialLoadRef.current = true;
    } catch (err) {
      console.error(`Error fetching ${errorLabel}:`, err);
      setStatus('error');
    } finally {
      inFlightRef.current = false;
    }
  }, [jobID, keyProbe, includeKey, fallbackKeys, reloadInterval, errorLabel]);

  useEffect(() => {
    didInitialLoadRef.current = false;
    lastStepByKeyRef.current = {};
    setSeries({});
    setKeys([]);
    setStatus('idle');

    refreshMetrics();
  }, [jobID, refreshMetrics]);

  useEffect(() => {
    if (!reloadInterval) return;
    const interval = setInterval(() => {
      refreshMetrics();
    }, reloadInterval);
    return () => clearInterval(interval);
  }, [reloadInterval, refreshMetrics]);

  return { series, setSeries, keys, metricKeys, status, refreshMetrics };
}

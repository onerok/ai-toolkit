'use client';

import { Job } from '@prisma/client';
import useJobSystemMetrics, { MetricPoint } from '@/hooks/useJobSystemMetrics';
import { useMemo, useState, useEffect } from 'react';
import { ResponsiveContainer, LineChart, Line, XAxis, YAxis, Tooltip, CartesianGrid, Legend } from 'recharts';

interface Props {
  job: Job;
  defaultCollapsed?: boolean;
}

function formatNum(v: number) {
  if (!Number.isFinite(v)) return '';
  if (Math.abs(v) >= 1000) return v.toFixed(0);
  if (Math.abs(v) >= 10) return v.toFixed(1);
  if (Math.abs(v) >= 1) return v.toFixed(2);
  return v.toPrecision(3);
}

// Metric display configuration
const METRIC_CONFIG: Record<string, { label: string; color: string; unit: string; axis: 'memory' | 'percent' }> = {
  vram_gb: { label: 'VRAM', color: 'rgba(96,165,250,1)', unit: 'GB', axis: 'memory' },
  ram_gb: { label: 'RAM', color: 'rgba(52,211,153,1)', unit: 'GB', axis: 'memory' },
  cpu_percent: { label: 'CPU', color: 'rgba(251,191,36,1)', unit: '%', axis: 'percent' },
};

export default function JobSystemMetricsGraph({ job, defaultCollapsed = true }: Props) {
  const { series, metricKeys, hasMetrics, status, refreshMetrics } = useJobSystemMetrics(job.id, 2000);

  const [collapsed, setCollapsed] = useState(defaultCollapsed);

  // UI-only downsample for rendering speed
  const [plotStride, setPlotStride] = useState(1);

  // show only last N points in the chart (0 = all)
  const [windowSize, setWindowSize] = useState<number>(4000);

  // which metrics are enabled
  const [enabled, setEnabled] = useState<Record<string, boolean>>({});

  // keep enabled map in sync with discovered keys
  useEffect(() => {
    setEnabled(prev => {
      const next = { ...prev };
      for (const k of metricKeys) {
        if (next[k] === undefined) next[k] = true;
      }
      for (const k of Object.keys(next)) {
        if (!metricKeys.includes(k)) delete next[k];
      }
      return next;
    });
  }, [metricKeys]);

  const activeKeys = useMemo(() => metricKeys.filter(k => enabled[k] !== false), [metricKeys, enabled]);

  const chartData = useMemo(() => {
    const stride = Math.max(1, plotStride | 0);
    const map = new Map<number, any>();

    for (const key of activeKeys) {
      let pts: MetricPoint[] = series[key] ?? [];

      // Filter and stride
      pts = pts
        .filter(p => p.value !== null && Number.isFinite(p.value as number))
        .filter((_, idx) => idx % stride === 0);

      // Windowing
      if (windowSize > 0 && pts.length > windowSize) {
        pts = pts.slice(pts.length - windowSize);
      }

      for (const p of pts) {
        const row = map.get(p.step) ?? { step: p.step };
        row[key] = p.value;
        map.set(p.step, row);
      }
    }

    const arr = Array.from(map.values());
    arr.sort((a, b) => a.step - b.step);
    return arr;
  }, [series, activeKeys, plotStride, windowSize]);

  const hasData = chartData.length > 1;

  // Check which axes we need
  const needsMemoryAxis = activeKeys.some(k => METRIC_CONFIG[k]?.axis === 'memory');
  const needsPercentAxis = activeKeys.some(k => METRIC_CONFIG[k]?.axis === 'percent');

  // Calculate memory domain
  const memoryDomain = useMemo((): [number, number | 'auto'] => {
    const memoryKeys = activeKeys.filter(k => METRIC_CONFIG[k]?.axis === 'memory');
    if (memoryKeys.length === 0) return [0, 'auto'];

    let max = 0;
    for (const row of chartData) {
      for (const k of memoryKeys) {
        const v = row[k];
        if (typeof v === 'number' && v > max) max = v;
      }
    }

    // Round up to nearest 4 GB for cleaner axis
    const roundedMax = Math.ceil(max / 4) * 4;
    return [0, roundedMax || 'auto'];
  }, [chartData, activeKeys]);

  // Latest values summary
  const latestValues = useMemo(() => {
    if (chartData.length === 0) return null;
    const last = chartData[chartData.length - 1];
    return {
      step: last.step,
      values: activeKeys.map(k => ({
        key: k,
        value: last[k] as number | undefined,
        config: METRIC_CONFIG[k],
      })),
    };
  }, [chartData, activeKeys]);

  // Don't render anything if no system metrics are being logged
  if (!hasMetrics && status === 'success') {
    return null;
  }

  return (
    <div className="bg-gray-900 rounded-xl shadow-lg overflow-hidden border border-gray-800 flex flex-col">
      <div
        className="bg-gray-800 px-4 py-3 flex items-center justify-between cursor-pointer"
        onClick={() => setCollapsed(c => !c)}
      >
        <div className="flex items-center gap-2">
          <div className="h-2 w-2 rounded-full bg-amber-400" />
          <h2 className="text-gray-100 text-sm font-medium">System metrics</h2>
          <span className="text-xs text-gray-400">
            {status === 'loading' && 'Loading...'}
            {status === 'refreshing' && 'Refreshing...'}
            {status === 'error' && 'Error'}
            {status === 'success' && hasData && `${chartData.length.toLocaleString()} steps`}
            {status === 'success' && !hasData && 'No data yet'}
          </span>
          {latestValues && (
            <span className="text-xs text-gray-500 ml-2">
              {latestValues.values
                .filter(v => v.value !== undefined)
                .map(v => `${v.config?.label}: ${formatNum(v.value!)}${v.config?.unit}`)
                .join(' | ')}
            </span>
          )}
        </div>

        <div className="flex items-center gap-2">
          <button
            type="button"
            onClick={e => {
              e.stopPropagation();
              refreshMetrics();
            }}
            className="px-3 py-1 rounded-md text-xs bg-gray-700/60 hover:bg-gray-700 text-gray-200 border border-gray-700"
          >
            Refresh
          </button>
          <svg
            className={`w-4 h-4 text-gray-400 transition-transform ${collapsed ? '' : 'rotate-180'}`}
            fill="none"
            viewBox="0 0 24 24"
            stroke="currentColor"
          >
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
          </svg>
        </div>
      </div>

      {!collapsed && (
        <>
          {/* Chart */}
          <div className="px-4 pt-4 pb-4">
            <div className="bg-gray-950 rounded-lg border border-gray-800 h-72 relative">
              {!hasData ? (
                <div className="h-full w-full flex items-center justify-center text-sm text-gray-400">
                  {status === 'error' ? 'Failed to load system metrics.' : 'Waiting for system metrics...'}
                </div>
              ) : (
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={chartData} margin={{ top: 10, right: needsPercentAxis ? 60 : 16, bottom: 10, left: 8 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" />
                    <XAxis
                      dataKey="step"
                      tick={{ fill: 'rgba(255,255,255,0.55)', fontSize: 12 }}
                      tickLine={{ stroke: 'rgba(255,255,255,0.15)' }}
                      axisLine={{ stroke: 'rgba(255,255,255,0.15)' }}
                      minTickGap={40}
                    />

                    {/* Left Y-axis for memory (GB) */}
                    {needsMemoryAxis && (
                      <YAxis
                        yAxisId="memory"
                        orientation="left"
                        tick={{ fill: 'rgba(255,255,255,0.55)', fontSize: 12 }}
                        tickLine={{ stroke: 'rgba(255,255,255,0.15)' }}
                        axisLine={{ stroke: 'rgba(255,255,255,0.15)' }}
                        width={50}
                        tickFormatter={v => `${formatNum(v)}`}
                        domain={memoryDomain}
                        label={{
                          value: 'GB',
                          angle: -90,
                          position: 'insideLeft',
                          style: { fill: 'rgba(255,255,255,0.4)', fontSize: 11 },
                        }}
                      />
                    )}

                    {/* Right Y-axis for percentage */}
                    {needsPercentAxis && (
                      <YAxis
                        yAxisId="percent"
                        orientation="right"
                        tick={{ fill: 'rgba(255,255,255,0.55)', fontSize: 12 }}
                        tickLine={{ stroke: 'rgba(255,255,255,0.15)' }}
                        axisLine={{ stroke: 'rgba(255,255,255,0.15)' }}
                        width={50}
                        tickFormatter={v => `${formatNum(v)}`}
                        domain={[0, 100]}
                        label={{
                          value: '%',
                          angle: 90,
                          position: 'insideRight',
                          style: { fill: 'rgba(255,255,255,0.4)', fontSize: 11 },
                        }}
                      />
                    )}

                    <Tooltip
                      cursor={{ stroke: 'rgba(251,191,36,0.25)', strokeWidth: 1 }}
                      contentStyle={{
                        background: 'rgba(17,24,39,0.96)',
                        border: '1px solid rgba(31,41,55,1)',
                        borderRadius: 10,
                        color: 'rgba(255,255,255,0.9)',
                        fontSize: 12,
                      }}
                      labelStyle={{ color: 'rgba(255,255,255,0.75)' }}
                      labelFormatter={(label: any) => `step ${label}`}
                      formatter={(value: any, name: any) => {
                        const config = METRIC_CONFIG[name];
                        return [`${formatNum(Number(value))} ${config?.unit ?? ''}`, config?.label ?? name];
                      }}
                    />

                    <Legend
                      wrapperStyle={{
                        paddingTop: 8,
                        color: 'rgba(255,255,255,0.7)',
                        fontSize: 12,
                      }}
                    />

                    {activeKeys.map(k => {
                      const config = METRIC_CONFIG[k];
                      if (!config) return null;

                      return (
                        <Line
                          key={k}
                          type="monotone"
                          dataKey={k}
                          name={k}
                          yAxisId={config.axis}
                          stroke={config.color}
                          strokeWidth={1.5}
                          dot={false}
                          isAnimationActive={false}
                        />
                      );
                    })}
                  </LineChart>
                </ResponsiveContainer>
              )}
            </div>
          </div>

          {/* Controls */}
          <div className="px-4 pb-3">
            <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
              <div className="bg-gray-950 border border-gray-800 rounded-lg p-3">
                <label className="block text-xs text-gray-400 mb-2">Series</label>
                {metricKeys.length === 0 ? (
                  <div className="text-sm text-gray-400">No metrics found yet.</div>
                ) : (
                  <div className="flex flex-wrap gap-2">
                    {metricKeys.map(k => {
                      const config = METRIC_CONFIG[k];
                      return (
                        <button
                          key={k}
                          type="button"
                          onClick={() => setEnabled(prev => ({ ...prev, [k]: !(prev[k] ?? true) }))}
                          className={[
                            'px-3 py-1 rounded-md text-xs border transition-colors',
                            enabled[k] === false
                              ? 'bg-gray-900 text-gray-400 border-gray-800 hover:bg-gray-800/60'
                              : 'bg-gray-900 text-gray-200 border-gray-800 hover:bg-gray-800/60',
                          ].join(' ')}
                          aria-pressed={enabled[k] !== false}
                          title={k}
                        >
                          <span
                            className="inline-block h-2 w-2 rounded-full mr-2"
                            style={{ background: config?.color ?? '#888' }}
                          />
                          {config?.label ?? k}
                        </button>
                      );
                    })}
                  </div>
                )}
              </div>

              <div className="bg-gray-950 border border-gray-800 rounded-lg p-3">
                <div className="flex items-center justify-between mb-1">
                  <label className="block text-xs text-gray-400">Plot stride</label>
                  <span className="text-xs text-gray-300">every {plotStride} pt</span>
                </div>
                <input
                  type="range"
                  min={1}
                  max={20}
                  value={plotStride}
                  onChange={e => setPlotStride(Number(e.target.value))}
                  className="w-full accent-amber-500"
                />
              </div>

              <div className="bg-gray-950 border border-gray-800 rounded-lg p-3">
                <div className="flex items-center justify-between mb-1">
                  <label className="block text-xs text-gray-400">Window (last N)</label>
                  <span className="text-xs text-gray-300">{windowSize === 0 ? 'all' : windowSize.toLocaleString()}</span>
                </div>
                <input
                  type="range"
                  min={0}
                  max={20000}
                  step={250}
                  value={windowSize}
                  onChange={e => setWindowSize(Number(e.target.value))}
                  className="w-full accent-amber-500"
                />
              </div>
            </div>
          </div>
        </>
      )}
    </div>
  );
}

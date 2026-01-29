import React, { useEffect, useState, useCallback } from 'react';
import { apiClient } from '@/utils/api';

type ServiceState = 'disabled' | 'stopped' | 'cold' | 'loading' | 'ready' | 'error';

interface CaptionServiceStatusProps {
  className?: string;
}

const STATE_CONFIG: Record<ServiceState, { color: string; label: string; pulse?: boolean }> = {
  disabled: { color: 'bg-gray-500', label: 'Disabled' },
  stopped: { color: 'bg-gray-500', label: 'Stopped' },
  cold: { color: 'bg-blue-500', label: 'Ready (Cold)' },
  loading: { color: 'bg-yellow-500', label: 'Loading Model...', pulse: true },
  ready: { color: 'bg-green-500', label: 'Ready' },
  error: { color: 'bg-red-500', label: 'Error' },
};

const STORAGE_KEY = 'caption-service-enabled';

export default function CaptionServiceStatus({ className = '' }: CaptionServiceStatusProps) {
  const [enabled, setEnabled] = useState<boolean>(() => {
    if (typeof window !== 'undefined') {
      const stored = localStorage.getItem(STORAGE_KEY);
      return stored !== 'false'; // Default to enabled
    }
    return true;
  });
  const [serviceState, setServiceState] = useState<ServiceState>('stopped');
  const [isToggling, setIsToggling] = useState(false);

  const fetchStatus = useCallback(async () => {
    if (!enabled) {
      setServiceState('disabled');
      return;
    }

    try {
      const response = await apiClient.get('/api/caption/service/status');
      const data = response.data;

      if (data.running) {
        setServiceState(data.modelState as ServiceState);
      } else {
        setServiceState('stopped');
      }
    } catch (error) {
      setServiceState('stopped');
    }
  }, [enabled]);

  // Start service when enabled
  const startService = useCallback(async () => {
    try {
      const response = await apiClient.post('/api/caption/service/start');
      console.log('[CaptionServiceStatus] Start response:', response.data);
      if (!response.data.success && response.data.error) {
        console.error('[CaptionServiceStatus] Start error:', response.data.error);
        setServiceState('error');
        return;
      }
      await fetchStatus();
    } catch (error: any) {
      console.error('[CaptionServiceStatus] Failed to start caption service:', error);
      console.error('[CaptionServiceStatus] Error details:', error?.response?.data);
      setServiceState('error');
    }
  }, [fetchStatus]);

  // Stop service when disabled
  const stopService = useCallback(async () => {
    try {
      await apiClient.post('/api/caption/service/stop');
      setServiceState('stopped');
    } catch (error) {
      console.error('Failed to stop caption service:', error);
    }
  }, []);

  // Toggle handler
  const handleToggle = async () => {
    setIsToggling(true);
    const newEnabled = !enabled;

    // Save preference
    localStorage.setItem(STORAGE_KEY, String(newEnabled));
    setEnabled(newEnabled);

    if (newEnabled) {
      await startService();
    } else {
      await stopService();
      setServiceState('disabled');
    }

    setIsToggling(false);
  };

  // Initial setup and polling
  useEffect(() => {
    if (enabled) {
      // Start service on mount if enabled
      startService();

      // Poll for status updates
      const interval = setInterval(fetchStatus, 3000);
      return () => clearInterval(interval);
    } else {
      setServiceState('disabled');
    }
  }, [enabled, startService, fetchStatus]);

  const config = STATE_CONFIG[serviceState];

  return (
    <div className={`flex items-center gap-3 ${className}`}>
      <div className="flex items-center gap-2">
        <span
          className={`inline-block h-2 w-2 rounded-full ${config.color} ${config.pulse ? 'animate-pulse' : ''}`}
          title={config.label}
        />
        <span className="text-xs text-gray-400">{config.label}</span>
      </div>

      <button
        onClick={handleToggle}
        disabled={isToggling}
        className={`
          relative inline-flex h-5 w-9 items-center rounded-full transition-colors
          ${enabled ? 'bg-blue-600' : 'bg-gray-600'}
          ${isToggling ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer hover:opacity-90'}
        `}
        title={enabled ? 'Disable auto-caption service' : 'Enable auto-caption service'}
      >
        <span
          className={`
            inline-block h-3 w-3 transform rounded-full bg-white transition-transform
            ${enabled ? 'translate-x-5' : 'translate-x-1'}
          `}
        />
      </button>
      <span className="text-xs text-gray-500">Auto-Caption</span>
    </div>
  );
}

/**
 * Hook to check if caption service is enabled and ready.
 */
export function useCaptionServiceEnabled(): boolean {
  const [enabled, setEnabled] = useState(false);

  useEffect(() => {
    if (typeof window !== 'undefined') {
      const stored = localStorage.getItem(STORAGE_KEY);
      setEnabled(stored !== 'false');
    }
  }, []);

  return enabled;
}

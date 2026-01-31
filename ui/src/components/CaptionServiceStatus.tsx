import React, { useEffect, useState, useCallback } from 'react';
import { apiClient } from '@/utils/api';

type ServiceState = 'disabled' | 'stopped' | 'cold' | 'loading' | 'ready' | 'error';
type CaptionProvider = 'local' | 'openai';

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

const STORAGE_KEY_ENABLED = 'caption-service-enabled';
const STORAGE_KEY_PROVIDER = 'caption-provider';

export default function CaptionServiceStatus({ className = '' }: CaptionServiceStatusProps) {
  const [provider, setProvider] = useState<CaptionProvider>(() => {
    if (typeof window !== 'undefined') {
      const stored = localStorage.getItem(STORAGE_KEY_PROVIDER);
      return (stored as CaptionProvider) || 'local';
    }
    return 'local';
  });

  const [enabled, setEnabled] = useState<boolean>(() => {
    if (typeof window !== 'undefined') {
      const stored = localStorage.getItem(STORAGE_KEY_ENABLED);
      return stored !== 'false';
    }
    return true;
  });

  const [serviceState, setServiceState] = useState<ServiceState>('stopped');
  const [isToggling, setIsToggling] = useState(false);

  const fetchStatus = useCallback(async () => {
    if (provider !== 'local' || !enabled) {
      setServiceState(provider === 'openai' ? 'ready' : 'disabled');
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
  }, [enabled, provider]);

  const startService = useCallback(async () => {
    if (provider !== 'local') return;

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
  }, [fetchStatus, provider]);

  const stopService = useCallback(async () => {
    try {
      await apiClient.post('/api/caption/service/stop');
      setServiceState('stopped');
    } catch (error) {
      console.error('Failed to stop caption service:', error);
    }
  }, []);

  // Handle provider change
  const handleProviderChange = async (newProvider: CaptionProvider) => {
    localStorage.setItem(STORAGE_KEY_PROVIDER, newProvider);
    setProvider(newProvider);

    if (newProvider === 'openai') {
      // Stop local service if running
      await stopService();
      setServiceState('ready');
    } else if (newProvider === 'local' && enabled) {
      // Start local service
      await startService();
    }
  };

  // Handle enable/disable toggle (only for local provider)
  const handleToggle = async () => {
    if (provider !== 'local') return;

    setIsToggling(true);
    const newEnabled = !enabled;

    localStorage.setItem(STORAGE_KEY_ENABLED, String(newEnabled));
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
    if (provider === 'local' && enabled) {
      startService();
      const interval = setInterval(fetchStatus, 3000);
      return () => clearInterval(interval);
    } else if (provider === 'openai') {
      setServiceState('ready');
    } else {
      setServiceState('disabled');
    }
  }, [enabled, provider, startService, fetchStatus]);

  const config = STATE_CONFIG[serviceState];
  const isLocalProvider = provider === 'local';

  return (
    <div className={`flex items-center gap-3 ${className}`}>
      {/* Provider selector */}
      <select
        value={provider}
        onChange={e => handleProviderChange(e.target.value as CaptionProvider)}
        className="text-xs bg-gray-800 text-gray-300 border border-gray-700 rounded px-2 py-1 focus:outline-none focus:border-gray-600"
      >
        <option value="local">Local VLM</option>
        <option value="openai">OpenAI</option>
      </select>

      {/* Status indicator */}
      <div className="flex items-center gap-2">
        <span
          className={`inline-block h-2 w-2 rounded-full ${config.color} ${config.pulse ? 'animate-pulse' : ''}`}
          title={config.label}
        />
        <span className="text-xs text-gray-400">{config.label}</span>
      </div>

      {/* Toggle (only for local provider) */}
      {isLocalProvider && (
        <button
          onClick={handleToggle}
          disabled={isToggling}
          className={`
            relative inline-flex h-5 w-9 items-center rounded-full transition-colors
            ${enabled ? 'bg-blue-600' : 'bg-gray-600'}
            ${isToggling ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer hover:opacity-90'}
          `}
          title={enabled ? 'Disable local caption service' : 'Enable local caption service'}
        >
          <span
            className={`
              inline-block h-3 w-3 transform rounded-full bg-white transition-transform
              ${enabled ? 'translate-x-5' : 'translate-x-1'}
            `}
          />
        </button>
      )}
    </div>
  );
}

/**
 * Get the current caption provider from localStorage.
 */
export function getCaptionProvider(): CaptionProvider {
  if (typeof window === 'undefined') return 'local';
  return (localStorage.getItem(STORAGE_KEY_PROVIDER) as CaptionProvider) || 'local';
}

/**
 * Check if caption service is enabled (local provider) or always ready (OpenAI).
 */
export function isCaptionEnabled(): boolean {
  if (typeof window === 'undefined') return false;
  const provider = getCaptionProvider();
  if (provider === 'openai') return true;
  return localStorage.getItem(STORAGE_KEY_ENABLED) !== 'false';
}

'use client';

import { useEffect, useState } from 'react';
import { apiClient } from '@/utils/api';

export default function useOutputList() {
  const [outputs, setOutputs] = useState<string[]>([]);
  const [status, setStatus] = useState<'idle' | 'loading' | 'success' | 'error'>('idle');

  const refreshOutputs = () => {
    setStatus('loading');
    apiClient
      .get('/api/output/list')
      .then(res => res.data)
      .then(data => {
        console.log('Outputs:', data);
        setOutputs(data);
        setStatus('success');
      })
      .catch(error => {
        console.error('Error fetching outputs:', error);
        setStatus('error');
      });
  };

  useEffect(() => {
    refreshOutputs();
  }, []);

  return { outputs, setOutputs, status, refreshOutputs };
}

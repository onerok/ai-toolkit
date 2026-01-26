'use client';

import { useEffect, useState } from 'react';
import { apiClient } from '@/utils/api';

export default function useTemplateList() {
  const [templates, setTemplates] = useState<string[]>([]);
  const [status, setStatus] = useState<'idle' | 'loading' | 'success' | 'error'>('idle');

  const refreshTemplates = () => {
    setStatus('loading');
    apiClient
      .get('/api/templates/list')
      .then(res => res.data)
      .then(data => {
        console.log('Templates:', data);
        // sort alphabetically
        data.sort((a: string, b: string) => a.localeCompare(b));
        setTemplates(data);
        setStatus('success');
      })
      .catch(error => {
        console.error('Error fetching templates:', error);
        setStatus('error');
      });
  };

  useEffect(() => {
    refreshTemplates();
  }, []);

  return { templates, setTemplates, status, refreshTemplates };
}

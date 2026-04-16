'use client';

import { useEffect, useState, useMemo } from 'react';
import { useRouter } from 'next/navigation';
import { TopBar, MainContent } from '@/components/layout';
import { SelectInput, TextInput } from '@/components/formInputs';
import { Button } from '@headlessui/react';
import useTemplateList from '@/hooks/useTemplateList';
import useDatasetList from '@/hooks/useDatasetList';
import useOutputList from '@/hooks/useOutputList';
import useGPUInfo from '@/hooks/useGPUInfo';
import { apiClient } from '@/utils/api';
import { FaCheck, FaSearch } from 'react-icons/fa';

export default function BatchQueuePage() {
  const router = useRouter();
  const { templates, status: templatesStatus } = useTemplateList();
  const { datasets, status: datasetsStatus } = useDatasetList();
  const { outputs, status: outputsStatus } = useOutputList();
  const { gpuList, isGPUInfoLoaded } = useGPUInfo();

  const [selectedTemplate, setSelectedTemplate] = useState<string>('');
  const [searchQuery, setSearchQuery] = useState<string>('');
  const [selectedDatasets, setSelectedDatasets] = useState<Set<string>>(new Set());
  const [gpuId, setGpuId] = useState<string>('0');
  const [submitting, setSubmitting] = useState(false);

  // Set default template when templates are loaded
  useEffect(() => {
    if (templatesStatus === 'success' && templates.length > 0 && !selectedTemplate) {
      setSelectedTemplate(templates[0]);
    }
  }, [templates, templatesStatus, selectedTemplate]);

  // Set default GPU when GPU info is loaded
  useEffect(() => {
    if (isGPUInfoLoaded && gpuList.length > 0) {
      setGpuId(`${gpuList[0].index}`);
    }
  }, [gpuList, isGPUInfoLoaded]);

  // Filter datasets that don't have output folders for the selected template
  const availableDatasets = useMemo(() => {
    if (!selectedTemplate || datasetsStatus !== 'success' || outputsStatus !== 'success') {
      return [];
    }

    // Filter out datasets that start with special characters (todo markers)
    const validDatasets = datasets.filter(dataset => !dataset.startsWith('!'));

    // Filter out datasets that already have output for this template
    return validDatasets.filter(dataset => {
      // Check if any output folder matches the pattern: {template}_{dataset}_*
      const hasOutput = outputs.some(output => {
        const pattern = `${selectedTemplate}_${dataset}_`;
        return output.startsWith(pattern);
      });
      return !hasOutput;
    });
  }, [selectedTemplate, datasets, outputs, datasetsStatus, outputsStatus]);

  // Filter by search query
  const filteredDatasets = useMemo(() => {
    if (!searchQuery.trim()) {
      return availableDatasets;
    }
    const query = searchQuery.toLowerCase();
    return availableDatasets.filter(dataset => dataset.toLowerCase().includes(query));
  }, [availableDatasets, searchQuery]);

  // Toggle dataset selection
  const toggleDataset = (dataset: string) => {
    const newSelected = new Set(selectedDatasets);
    if (newSelected.has(dataset)) {
      newSelected.delete(dataset);
    } else {
      newSelected.add(dataset);
    }
    setSelectedDatasets(newSelected);
  };

  // Select all visible datasets
  const selectAll = () => {
    const newSelected = new Set(selectedDatasets);
    filteredDatasets.forEach(dataset => newSelected.add(dataset));
    setSelectedDatasets(newSelected);
  };

  // Deselect all
  const deselectAll = () => {
    setSelectedDatasets(new Set());
  };

  // Submit selected datasets to queue
  const handleSubmit = async () => {
    if (selectedDatasets.size === 0) {
      alert('Please select at least one dataset');
      return;
    }

    setSubmitting(true);
    try {
      const response = await apiClient.post('/api/batch-queue', {
        template: selectedTemplate,
        datasets: Array.from(selectedDatasets),
        gpu_ids: gpuId,
      });

      const data = response.data;

      if (data.errors > 0) {
        alert(`Created ${data.created} jobs. ${data.errors} failed:\n${data.errorDetails.map((e: any) => `${e.dataset}: ${e.error}`).join('\n')}`);
      } else {
        alert(`Successfully created ${data.created} jobs!`);
      }

      // Clear selection and redirect to jobs-extended page
      setSelectedDatasets(new Set());
      router.push('/jobs-extended');
    } catch (error: any) {
      console.error('Error submitting batch jobs:', error);
      alert('Failed to create jobs. Please try again.');
    } finally {
      setSubmitting(false);
    }
  };

  const isLoading = templatesStatus === 'loading' || datasetsStatus === 'loading' || outputsStatus === 'loading';

  return (
    <>
      <TopBar>
        <div className="flex items-center gap-4 w-full">
          <h1 className="text-lg font-semibold whitespace-nowrap">From Dataset to Queue</h1>

          <div className="flex items-center gap-2">
            <SelectInput
              value={selectedTemplate}
              onChange={setSelectedTemplate}
              options={templates.map(t => ({ value: t, label: t }))}
              className="w-40"
            />

            <SelectInput
              value={gpuId}
              onChange={setGpuId}
              options={gpuList.map((gpu: any) => ({ value: `${gpu.index}`, label: `GPU #${gpu.index}` }))}
              className="w-32"
            />
          </div>

          <div className="flex-1"></div>

          <div className="flex items-center gap-2">
            <span className="text-sm text-gray-400">
              {selectedDatasets.size} selected
            </span>
            <Button
              onClick={handleSubmit}
              disabled={submitting || selectedDatasets.size === 0}
              className="px-4 py-2 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 disabled:cursor-not-allowed text-white rounded-lg transition-colors"
            >
              {submitting ? 'Adding...' : 'Add to Queue'}
            </Button>
          </div>
        </div>
      </TopBar>

      <MainContent>
        {isLoading ? (
          <div className="flex items-center justify-center h-64">
            <div className="text-gray-400">Loading...</div>
          </div>
        ) : (
          <div className="space-y-4">
            {/* Search and actions bar */}
            <div className="flex items-center gap-4">
              <div className="relative flex-1 max-w-md">
                <FaSearch className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-500" />
                <input
                  type="text"
                  placeholder="Search datasets..."
                  value={searchQuery}
                  onChange={e => setSearchQuery(e.target.value)}
                  className="w-full pl-10 pr-4 py-2 bg-gray-800 border border-gray-700 rounded-lg focus:ring-2 focus:ring-blue-600 focus:border-transparent"
                />
              </div>

              <div className="flex items-center gap-2">
                <Button
                  onClick={selectAll}
                  className="px-3 py-2 text-sm bg-gray-700 hover:bg-gray-600 text-white rounded-lg transition-colors"
                >
                  Select All ({filteredDatasets.length})
                </Button>
                <Button
                  onClick={deselectAll}
                  className="px-3 py-2 text-sm bg-gray-700 hover:bg-gray-600 text-white rounded-lg transition-colors"
                >
                  Deselect All
                </Button>
              </div>
            </div>

            {/* Stats */}
            <div className="text-sm text-gray-400">
              Showing {filteredDatasets.length} datasets without {selectedTemplate} training output
              {searchQuery && ` (filtered from ${availableDatasets.length})`}
            </div>

            {/* Dataset list */}
            <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5 gap-2">
              {filteredDatasets.map(dataset => (
                <div
                  key={dataset}
                  onClick={() => toggleDataset(dataset)}
                  className={`flex items-center gap-3 p-3 rounded-lg cursor-pointer transition-colors ${
                    selectedDatasets.has(dataset)
                      ? 'bg-blue-600/30 border border-blue-500'
                      : 'bg-gray-800 border border-gray-700 hover:bg-gray-700'
                  }`}
                >
                  <div
                    className={`w-5 h-5 rounded border-2 flex items-center justify-center ${
                      selectedDatasets.has(dataset)
                        ? 'bg-blue-600 border-blue-600'
                        : 'border-gray-500'
                    }`}
                  >
                    {selectedDatasets.has(dataset) && <FaCheck className="w-3 h-3 text-white" />}
                  </div>
                  <span className="text-sm truncate">{dataset}</span>
                </div>
              ))}
            </div>

            {filteredDatasets.length === 0 && (
              <div className="text-center py-12 text-gray-500">
                {availableDatasets.length === 0
                  ? `All datasets already have ${selectedTemplate} training output`
                  : 'No datasets match your search'}
              </div>
            )}
          </div>
        )}
      </MainContent>
    </>
  );
}

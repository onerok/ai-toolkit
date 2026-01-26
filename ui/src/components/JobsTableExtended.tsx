import { useMemo, useState, useEffect } from 'react';
import useJobsList from '@/hooks/useJobsList';
import Link from 'next/link';
import UniversalTable, { TableColumn } from '@/components/UniversalTable';
import { GpuInfo, JobConfig } from '@/types';
import JobActionBar from './JobActionBar';
import { Job, Queue } from '@prisma/client';
import useQueueList from '@/hooks/useQueueList';
import classNames from 'classnames';
import { startQueue, stopQueue } from '@/utils/queue';
import { CgSpinner } from 'react-icons/cg';
import useGPUInfo from '@/hooks/useGPUInfo';
import { apiClient } from '@/utils/api';
import { openConfirm } from '@/components/ConfirmModal';
import { loadStepsPerMinuteConfig, calculateQueueRemainingTime, formatRemainingTime } from '@/utils/timeEstimation';

interface JobsTableProps {
  autoStartQueue?: boolean;
  onlyActive?: boolean;
}

export default function JobsTableExtended({ onlyActive = false }: JobsTableProps) {
  const { jobs, status, refreshJobs } = useJobsList(onlyActive, 5000);
  const { queues, status: queueStatus, refreshQueues } = useQueueList();
  const { gpuList, isGPUInfoLoaded } = useGPUInfo();

  // New state for Advanced Queue features
  const [currentPage, setCurrentPage] = useState(1);
  const [pageSize, setPageSize] = useState<number | 'all'>(25);
  const [searchQuery, setSearchQuery] = useState('');
  const [statusFilter, setStatusFilter] = useState('all');

  // Load steps per minute config on mount
  const [configLoaded, setConfigLoaded] = useState(false);

  useEffect(() => {
    if (!configLoaded) {
      loadStepsPerMinuteConfig();
      setConfigLoaded(true);
    }
  }, [configLoaded]);

  const refresh = () => {
    refreshJobs();
    refreshQueues();
  };

  const columns: TableColumn[] = [
    {
      title: 'Name',
      key: 'name',
      render: row => (
        <Link href={`/jobs/${row.id}`} className="font-medium whitespace-nowrap">
          {['running', 'stopping'].includes(row.status) ? (
            <CgSpinner className="inline animate-spin mr-2 text-blue-400" />
          ) : null}
          {row.name}
        </Link>
      ),
    },
    {
      title: 'Steps',
      key: 'steps',
      render: row => {
        const jobConfig: JobConfig = JSON.parse(row.job_config);
        const totalSteps = jobConfig.config.process[0].train.steps;

        return (
          <div>
            <div className="text-xs text-gray-400">
              {row.step} / {totalSteps}
            </div>
            <div className="bg-gray-700 rounded-full h-1.5">
              <div
                className="bg-blue-500 h-1.5 rounded-full"
                style={{ width: `${(row.step / totalSteps) * 100}%` }}
              ></div>
            </div>
          </div>
        );
      },
    },
    {
      title: 'GPU',
      key: 'gpu_ids',
      render: row => (
        <span>{row.gpu_ids || 'Pending'}</span>
      )
    },
    {
      title: 'Status',
      key: 'status',
      render: row => {
        let statusClass = 'text-gray-400';
        if (row.status === 'completed') statusClass = 'text-green-400';
        if (row.status === 'failed') statusClass = 'text-red-400';
        if (row.status === 'running') statusClass = 'text-blue-400';

        return <span className={statusClass}>{row.status}</span>;
      },
    },
    {
      title: 'Info',
      key: 'info',
      className: 'truncate max-w-xs',
    },
    {
      title: 'Actions',
      key: 'actions',
      className: 'text-right',
      render: row => {
        return <JobActionBar job={row} onRefresh={refreshJobs} autoStartQueue={false} />;
      },
    },
  ];

  const jobsDict = useMemo(() => {
    if (!isGPUInfoLoaded) return {};
    if (jobs.length === 0) return {};
    let jd: { [key: string]: { name: string; jobs: Job[] } } = {};
    gpuList.forEach(gpu => {
      jd[`${gpu.index}`] = { name: `${gpu.name}`, jobs: [] };
    });
    jd['Idle'] = { name: 'Idle', jobs: [] };
    jobs.forEach(job => {
      const gpu = gpuList.find(gpu => job.gpu_ids?.split(',').includes(gpu.index.toString())) as GpuInfo;
      const key = `${gpu?.index || '0'}`;
      if (['queued', 'running', 'stopping'].includes(job.status) && key in jd) {
        jd[key].jobs.push(job);
      } else {
        jd['Idle'].jobs.push(job);
      }
    });
    // sort the queued/running jobs by queue position
    Object.keys(jd).forEach(key => {
      if (key === 'Idle') return;
      jd[key].jobs.sort((a, b) => {
        if (a.queue_position === null) return 1;
        if (b.queue_position === null) return -1;
        return a.queue_position - b.queue_position;
      });
    });
    return jd;
  }, [jobs, queues, isGPUInfoLoaded, gpuList]);

  // Filter and Paginate Idle jobs
  const idleJobs = jobsDict['Idle']?.jobs || [];

  const filteredIdleJobs = useMemo(() => {
    return idleJobs.filter(job => {
        const matchesSearch = job.name.toLowerCase().includes(searchQuery.toLowerCase());
        const matchesStatus = statusFilter === 'all' || job.status === statusFilter;
        return matchesSearch && matchesStatus;
    });
  }, [idleJobs, searchQuery, statusFilter]);

  // Reset page when filters change
  useEffect(() => {
    setCurrentPage(1);
  }, [searchQuery, statusFilter, pageSize]);

  const paginatedIdleJobs = useMemo(() => {
      if (pageSize === 'all') return filteredIdleJobs;
      const start = (currentPage - 1) * (pageSize as number);
      return filteredIdleJobs.slice(start, start + (pageSize as number));
  }, [filteredIdleJobs, currentPage, pageSize]);

  const totalPages = pageSize === 'all' ? 1 : Math.ceil(filteredIdleJobs.length / (pageSize as number));

  const availableStatuses = useMemo(() => {
      const statuses = new Set(idleJobs.map(job => job.status));
      return ['all', ...Array.from(statuses)];
  }, [idleJobs]);

  const pageSizeOptions = [25, 50, 100, 250, 500, 1000, 2000, 5000, 'all'];

  // Bulk action handlers
  const handleStopAllQueued = async () => {
    openConfirm({
      title: 'Stop All Queued Jobs',
      message: 'Are you sure you want to stop all queued jobs? This will mark them as stopped but leave running jobs intact.',
      type: 'warning',
      confirmText: 'Stop All Queued',
      onConfirm: async () => {
        try {
          const response = await apiClient.post('/api/jobs/bulk', { action: 'stop-queued' });
          if (response.data.success) {
            alert(response.data.message);
            refresh();
          }
        } catch (error) {
          console.error('Error stopping queued jobs:', error);
          alert('Failed to stop queued jobs');
        }
      },
    });
  };

  const handleResumeStopped = async () => {
    openConfirm({
      title: 'Resume Stopped Jobs',
      message: 'Are you sure you want to resume all stopped jobs? This will add them back to the queue.',
      type: 'info',
      confirmText: 'Resume All',
      onConfirm: async () => {
        try {
          const response = await apiClient.post('/api/jobs/bulk', { action: 'resume-stopped' });
          if (response.data.success) {
            alert(response.data.message);
            refresh();
          }
        } catch (error) {
          console.error('Error resuming stopped jobs:', error);
          alert('Failed to resume stopped jobs');
        }
      },
    });
  };

  let isLoading = status === 'loading' || queueStatus === 'loading' || !isGPUInfoLoaded;

  // if job dict is populated, we are always loaded
  if (Object.keys(jobsDict).length > 0) isLoading = false;

  // Calculate remaining time for each GPU queue
  const calculateGpuQueueRemainingTime = (gpuJobs: Job[]): string => {
    const totalMinutes = calculateQueueRemainingTime(gpuJobs);
    return formatRemainingTime(totalMinutes);
  };

  return (
    <div>
      {/* Bulk Action Buttons */}
      <div className="mb-4 flex gap-2">
        <button
          onClick={handleStopAllQueued}
          className="px-4 py-2 bg-red-700 hover:bg-red-600 text-white rounded-md text-sm font-medium transition-colors"
          title="Stop all queued jobs (leave running jobs intact)"
        >
          Stop All Queued Jobs
        </button>
        <button
          onClick={handleResumeStopped}
          className="px-4 py-2 bg-green-700 hover:bg-green-600 text-white rounded-md text-sm font-medium transition-colors"
          title="Resume all stopped jobs (add them back to queue)"
        >
          Resume Stopped Jobs
        </button>
      </div>
      {Object.keys(jobsDict)
        .sort()
        .filter(key => key !== 'Idle')
        .map(gpuKey => {
          const queue = queues.find(q => `${q.gpu_ids}` === gpuKey) as Queue;
          return (
            <div key={gpuKey} className="mb-6">
              <div
                className={classNames(
                  'text-md flex px-4 py-1 rounded-t-lg',
                  { 'bg-green-900': queue?.is_running },
                  { 'bg-red-900': !queue?.is_running },
                )}
              >
                <div className="flex items-center space-x-2 flex-1 py-2">
                  <h2 className="font-semibold text-gray-100">{jobsDict[gpuKey].name}</h2>
                  <span className="px-2 py-0.5 bg-gray-700 rounded-full text-xs text-gray-300"># {queue?.gpu_ids}</span>
                </div>
                <div className="text-sm text-gray-300 italic flex items-center">
                  {queue?.is_running ? (
                    <>
                      <span className="text-green-400 mr-2">Queue Running</span>
                      <span className="text-gray-400 mr-2">|</span>
                      <span className="text-blue-400 mr-2">
                        ETA: {calculateGpuQueueRemainingTime(jobsDict[gpuKey].jobs)}
                      </span>
                      <button
                        onClick={async () => {
                          await stopQueue(queue.gpu_ids as string);
                          refresh();
                        }}
                        className="ml-4 text-xs bg-red-900 hover:bg-red-800 px-2 py-1 rounded"
                      >
                        STOP
                      </button>
                    </>
                  ) : (
                    <>
                      <span className="text-red-400 mr-2">Queue Stopped</span>
                      <span className="text-gray-400 mr-2">|</span>
                      <span className="text-blue-400 mr-2">
                        ETA: {calculateGpuQueueRemainingTime(jobsDict[gpuKey].jobs)}
                      </span>
                      <button
                        onClick={async () => {
                          await startQueue(gpuKey);
                          refresh();
                        }}
                        className="ml-4 text-xs bg-green-700 hover:bg-green-600 px-2 py-1 rounded"
                      >
                        START
                      </button>
                    </>
                  )}
                </div>
              </div>
              <UniversalTable
                columns={columns}
                rows={jobsDict[gpuKey].jobs}
                isLoading={isLoading}
                onRefresh={refresh}
                theadClassName={queue?.is_running ? 'bg-green-950' : 'bg-red-950'}
              />
            </div>
          );
        })}
      {!onlyActive && Object.keys(jobsDict).includes('Idle') && (
        <div className="mb-6">
          <div className="text-md flex px-4 py-1 rounded-t-lg bg-slate-600 justify-between items-center">
            <div className="flex items-center space-x-2 py-2">
              <h2 className="font-semibold text-gray-100">Idle / History</h2>
            </div>
          </div>

          {/* Controls for Idle Section */}
          <div className="bg-gray-800 p-4 border-b border-gray-700 flex flex-wrap gap-4 items-center">
             {/* Search */}
             <div className="flex items-center gap-2">
                <label className="text-sm text-gray-400">Search:</label>
                <input
                    type="text"
                    value={searchQuery}
                    onChange={(e) => setSearchQuery(e.target.value)}
                    placeholder="Search job name..."
                    className="bg-gray-700 text-gray-200 px-2 py-1 rounded border border-gray-600 focus:outline-none focus:border-blue-500 text-sm"
                />
             </div>

             {/* Status Filter */}
             <div className="flex items-center gap-2">
                <label className="text-sm text-gray-400">Status:</label>
                <select
                    value={statusFilter}
                    onChange={(e) => setStatusFilter(e.target.value)}
                    className="bg-gray-700 text-gray-200 px-2 py-1 rounded border border-gray-600 focus:outline-none focus:border-blue-500 text-sm"
                >
                    {availableStatuses.map(s => (
                        <option key={s} value={s} className="capitalize">{s}</option>
                    ))}
                </select>
             </div>

             {/* Page Size */}
             <div className="flex items-center gap-2">
                <label className="text-sm text-gray-400">Show:</label>
                <select
                    value={pageSize}
                    onChange={(e) => setPageSize(e.target.value === 'all' ? 'all' : Number(e.target.value))}
                    className="bg-gray-700 text-gray-200 px-2 py-1 rounded border border-gray-600 focus:outline-none focus:border-blue-500 text-sm"
                >
                    {pageSizeOptions.map(opt => (
                        <option key={opt} value={opt}>{opt}</option>
                    ))}
                </select>
             </div>

             {/* Pagination Info */}
             <div className="ml-auto text-sm text-gray-400">
                Showing {paginatedIdleJobs.length} of {filteredIdleJobs.length}
             </div>
          </div>

          <UniversalTable columns={columns} rows={paginatedIdleJobs} isLoading={isLoading} onRefresh={refresh} />

          {/* Pagination Controls */}
          {pageSize !== 'all' && totalPages > 1 && (
              <div className="bg-gray-800 p-3 rounded-b-lg flex justify-center items-center gap-2">
                  <button
                    onClick={() => setCurrentPage(p => Math.max(1, p - 1))}
                    disabled={currentPage === 1}
                    className="px-3 py-1 bg-gray-700 hover:bg-gray-600 disabled:opacity-50 disabled:hover:bg-gray-700 rounded text-sm text-gray-200"
                  >
                    Prev
                  </button>
                  <span className="text-sm text-gray-300">
                    Page {currentPage} of {totalPages}
                  </span>
                  <button
                    onClick={() => setCurrentPage(p => Math.min(totalPages, p + 1))}
                    disabled={currentPage === totalPages}
                    className="px-3 py-1 bg-gray-700 hover:bg-gray-600 disabled:opacity-50 disabled:hover:bg-gray-700 rounded text-sm text-gray-200"
                  >
                    Next
                  </button>
              </div>
          )}
        </div>
      )}
    </div>
  );
}

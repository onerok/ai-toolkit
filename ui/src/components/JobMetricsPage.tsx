'use client';

import { Job } from '@prisma/client';
import JobLossGraph from './JobLossGraph';
import JobSystemMetricsGraph from './JobSystemMetricsGraph';

interface Props {
  job: Job;
}

export default function JobMetricsPage({ job }: Props) {
  return (
    <div className="flex flex-col gap-4">
      <JobLossGraph job={job} />
      <JobSystemMetricsGraph job={job} defaultCollapsed={true} />
    </div>
  );
}

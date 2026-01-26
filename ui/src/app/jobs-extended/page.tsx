'use client';

import JobsTableExtended from '@/components/JobsTableExtended';
import { TopBar, MainContent } from '@/components/layout';
import Link from 'next/link';

export default function AdvancedQueue() {
  return (
    <>
      <TopBar>
        <div>
          <h1 className="text-lg">Advanced Queue</h1>
        </div>
        <div className="flex-1"></div>
        <div>
          <Link href="/jobs/new" className="text-gray-200 bg-slate-600 px-3 py-1 rounded-md">
            New Training Job
          </Link>
        </div>
      </TopBar>
      <MainContent>
        <JobsTableExtended />
      </MainContent>
    </>
  );
}

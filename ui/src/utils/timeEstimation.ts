import { Job } from '@prisma/client';
import { JobConfig } from '@/types';
import jobPerformanceConfig from '@/config/jobPerformance.json';

export interface StepsPerMinuteConfig {
  [key: string]: number;
}

let stepsPerMinuteCache: StepsPerMinuteConfig = jobPerformanceConfig.stepsPerMinute;

/**
 * Load the steps per minute configuration
 */
export function loadStepsPerMinuteConfig(): StepsPerMinuteConfig {
  return stepsPerMinuteCache;
}

/**
 * Get the job type prefix from job name
 */
export function getJobTypePrefix(jobName: string): string | null {
  const prefixes = Object.keys(stepsPerMinuteCache);
  for (const prefix of prefixes) {
    if (jobName.startsWith(prefix)) {
      return prefix;
    }
  }
  return null;
}

/**
 * Get steps per minute for a specific job based on its name
 */
export function getStepsPerMinute(jobName: string): number {
  const prefix = getJobTypePrefix(jobName);
  if (prefix && stepsPerMinuteCache[prefix]) {
    return stepsPerMinuteCache[prefix];
  }
  // Default fallback if no prefix matches
  return 60;
}

/**
 * Calculate remaining steps for a job
 */
export function getRemainingSteps(job: Job): number {
  try {
    const jobConfig: JobConfig = JSON.parse(job.job_config);
    const totalSteps = jobConfig.config.process[0].train.steps;
    const remainingSteps = totalSteps - job.step;
    return Math.max(0, remainingSteps);
  } catch (error) {
    console.error('Error calculating remaining steps:', error);
    return 0;
  }
}

/**
 * Calculate remaining time in minutes for a single job
 */
export function calculateJobRemainingTime(job: Job): number {
  const remainingSteps = getRemainingSteps(job);
  const stepsPerMinute = getStepsPerMinute(job.name);

  if (stepsPerMinute === 0) return 0;

  return remainingSteps / stepsPerMinute;
}

/**
 * Calculate total remaining time for an array of jobs
 */
export function calculateQueueRemainingTime(jobs: Job[]): number {
  let totalMinutes = 0;

  for (const job of jobs) {
    if (job.status === 'queued' || job.status === 'running') {
      totalMinutes += calculateJobRemainingTime(job);
    }
  }

  return totalMinutes;
}

/**
 * Format minutes into a human-readable string like "3 days 12 hours 44 minutes"
 */
export function formatRemainingTime(totalMinutes: number): string {
  if (totalMinutes <= 0) {
    return '0 minutes';
  }

  const days = Math.floor(totalMinutes / (24 * 60));
  const hours = Math.floor((totalMinutes % (24 * 60)) / 60);
  const minutes = Math.floor(totalMinutes % 60);

  const parts: string[] = [];

  if (days > 0) {
    parts.push(`${days} day${days !== 1 ? 's' : ''}`);
  }
  if (hours > 0) {
    parts.push(`${hours} hour${hours !== 1 ? 's' : ''}`);
  }
  if (minutes > 0 || parts.length === 0) {
    parts.push(`${minutes} minute${minutes !== 1 ? 's' : ''}`);
  }

  return parts.join(' ');
}

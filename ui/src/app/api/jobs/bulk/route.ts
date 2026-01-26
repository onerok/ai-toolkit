import { NextResponse } from 'next/server';
import { PrismaClient } from '@prisma/client';

const prisma = new PrismaClient();

// Stop all queued jobs (mark as stopped, leave running ones intact)
export async function POST(request: Request) {
  try {
    const body = await request.json();
    const { action } = body;

    if (action === 'stop-queued') {
      // Mark all queued jobs as stopped (leave running ones intact)
      const result = await prisma.job.updateMany({
        where: {
          status: 'queued',
        },
        data: {
          status: 'stopped',
          stop: true,
          info: 'Job stopped',
        },
      });

      return NextResponse.json({
        success: true,
        count: result.count,
        message: `Stopped ${result.count} queued job(s)`
      });
    }

    if (action === 'resume-stopped') {
      // Find all stopped jobs
      const stoppedJobs = await prisma.job.findMany({
        where: {
          status: 'stopped',
        },
      });

      if (stoppedJobs.length === 0) {
        return NextResponse.json({
          success: true,
          count: 0,
          message: 'No stopped jobs to resume'
        });
      }

      // Get the highest queue position
      const highestQueuePosition = await prisma.job.aggregate({
        _max: {
          queue_position: true,
        },
      });
      let currentQueuePosition = (highestQueuePosition._max.queue_position || 0) + 1000;

      // Update each stopped job to queued with proper queue positions
      let updated = 0;
      for (const job of stoppedJobs) {
        // Make sure the queue exists for this GPU
        const queue = await prisma.queue.findFirst({
          where: { gpu_ids: job.gpu_ids },
        });

        if (!queue) {
          await prisma.queue.create({
            data: {
              gpu_ids: job.gpu_ids,
              is_running: false,
            },
          });
        }

        await prisma.job.update({
          where: { id: job.id },
          data: {
            status: 'queued',
            stop: false,
            return_to_queue: false,
            queue_position: currentQueuePosition,
            info: 'Job queued',
          },
        });
        currentQueuePosition += 1000;
        updated++;
      }

      return NextResponse.json({
        success: true,
        count: updated,
        message: `Resumed ${updated} stopped job(s)`
      });
    }

    return NextResponse.json({ error: 'Invalid action' }, { status: 400 });
  } catch (error) {
    console.error('Error in bulk job operation:', error);
    return NextResponse.json({ error: 'Failed to perform bulk operation' }, { status: 500 });
  }
}

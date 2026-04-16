import { NextResponse } from 'next/server';
import { PrismaClient } from '@prisma/client';
import fs from 'fs';
import path from 'path';
import YAML from 'yaml';

const prisma = new PrismaClient();

export async function POST(request: Request) {
  try {
    const body = await request.json();
    const { template, datasets, gpu_ids } = body;

    if (!template || !datasets || !Array.isArray(datasets) || datasets.length === 0) {
      return NextResponse.json(
        { error: 'Template and datasets array are required' },
        { status: 400 }
      );
    }

    // Read the template file
    const templatesPath = path.resolve(process.cwd(), '..', 'templates');
    const templateFile = path.join(templatesPath, `${template}_template.yaml`);

    if (!fs.existsSync(templateFile)) {
      return NextResponse.json({ error: 'Template not found' }, { status: 404 });
    }

    const templateContent = fs.readFileSync(templateFile, 'utf-8');

    // Get the current highest queue position
    const highestQueuePosition = await prisma.job.aggregate({
      _max: {
        queue_position: true,
      },
    });
    let currentQueuePosition = ((highestQueuePosition._max?.queue_position) || 0) + 1000;

    const createdJobs = [];
    const errors = [];

    for (const datasetName of datasets) {
      try {
        // Replace [dataset_name] placeholder in the template
        const jobConfigYaml = templateContent.replace(/\[dataset_name\]/g, datasetName);

        // Parse the YAML to get the job config object
        const jobConfig = YAML.parse(jobConfigYaml);

        // Extract the job name from the config
        const jobName = (jobConfig as any).config?.name || `${template}_${datasetName}_v1`;

        // Create the job in the database
        const job = await prisma.job.create({
          data: {
            name: jobName,
            gpu_ids: gpu_ids || '0',
            job_config: JSON.stringify(jobConfig),
            queue_position: currentQueuePosition,
          },
        });

        // Start the job (set status to 'queued' and ensure queue exists)
        const jobGpuIds = gpu_ids || '0';

        // Make sure the queue exists for this GPU
        const queue = await prisma.queue.findFirst({
          where: { gpu_ids: jobGpuIds },
        });

        if (!queue) {
          await prisma.queue.create({
            data: {
              gpu_ids: jobGpuIds,
              is_running: false,
            },
          });
        }

        // Update job status to queued
        await prisma.job.update({
          where: { id: job.id },
          data: {
            status: 'queued',
            stop: false,
            return_to_queue: false,
            info: 'Job queued',
          },
        });

        createdJobs.push(job);
        currentQueuePosition += 1000;
      } catch (error: any) {
        if (error.code === 'P2002') {
          // Unique constraint violation - job name already exists
          errors.push({ dataset: datasetName, error: 'Job name already exists' });
        } else {
          errors.push({ dataset: datasetName, error: error.message });
        }
      }
    }

    return NextResponse.json({
      created: createdJobs.length,
      errors: errors.length,
      jobs: createdJobs,
      errorDetails: errors,
    });
  } catch (error: any) {
    console.error('Error creating batch jobs:', error);
    return NextResponse.json({ error: 'Failed to create batch jobs' }, { status: 500 });
  }
}

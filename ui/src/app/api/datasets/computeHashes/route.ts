import { NextResponse } from 'next/server';
import fs from 'fs';
import path from 'path';
import { getDatasetsRoot } from '@/server/settings';
import { getOrComputeHashes } from '@/server/phash';

const IMAGE_EXTENSIONS = ['.png', '.jpg', '.jpeg', '.webp'];

function findImagesRecursively(dir: string): string[] {
  let results: string[] = [];
  const items = fs.readdirSync(dir);

  for (const item of items) {
    const itemPath = path.join(dir, item);
    const stat = fs.statSync(itemPath);

    if (stat.isDirectory() && item !== '_controls' && !item.startsWith('.')) {
      results = results.concat(findImagesRecursively(itemPath));
    } else {
      const ext = path.extname(itemPath).toLowerCase();
      if (IMAGE_EXTENSIONS.includes(ext)) {
        results.push(itemPath);
      }
    }
  }

  return results;
}

export async function POST(request: Request) {
  const datasetsPath = await getDatasetsRoot();
  const body = await request.json();
  const { datasetName } = body;
  const datasetFolder = path.join(datasetsPath, datasetName);

  try {
    if (!fs.existsSync(datasetFolder)) {
      return NextResponse.json({ error: `Folder '${datasetName}' not found` }, { status: 404 });
    }

    const imagePaths = findImagesRecursively(datasetFolder);
    const hashes = await getOrComputeHashes(datasetFolder, imagePaths);

    return NextResponse.json({ hashes });
  } catch (error) {
    console.error('Error computing hashes:', error);
    return NextResponse.json({ error: 'Failed to compute hashes' }, { status: 500 });
  }
}

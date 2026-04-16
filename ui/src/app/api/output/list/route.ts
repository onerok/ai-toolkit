import { NextResponse } from 'next/server';
import fs from 'fs';
import { getTrainingFolder } from '@/server/settings';

export async function GET() {
  try {
    const outputPath = await getTrainingFolder();

    // if folder doesn't exist, return empty array
    if (!fs.existsSync(outputPath)) {
      return NextResponse.json([]);
    }

    // find all folders in the output directory
    const folders = fs
      .readdirSync(outputPath, { withFileTypes: true })
      .filter(dirent => dirent.isDirectory())
      .filter(dirent => !dirent.name.startsWith('.'))
      .map(dirent => dirent.name);

    return NextResponse.json(folders);
  } catch (error) {
    console.error('Error listing output folders:', error);
    return NextResponse.json({ error: 'Failed to fetch output folders' }, { status: 500 });
  }
}

import { NextResponse } from 'next/server';
import fs from 'fs';
import path from 'path';

export async function GET() {
  try {
    // Templates folder is at the root of ai-toolkit (parent of ui folder)
    const templatesPath = path.resolve(process.cwd(), '..', 'templates');

    // if folder doesn't exist, return empty array
    if (!fs.existsSync(templatesPath)) {
      return NextResponse.json([]);
    }

    // find all *_template.yaml files and extract the template name
    const templates = fs
      .readdirSync(templatesPath, { withFileTypes: true })
      .filter(dirent => dirent.isFile())
      .filter(dirent => dirent.name.endsWith('_template.yaml'))
      .map(dirent => dirent.name.replace('_template.yaml', ''));

    return NextResponse.json(templates);
  } catch (error) {
    console.error('Error listing templates:', error);
    return NextResponse.json({ error: 'Failed to fetch templates' }, { status: 500 });
  }
}

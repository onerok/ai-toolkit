import { NextResponse } from 'next/server';
import fs from 'fs';
import path from 'path';

export async function GET(request: Request, { params }: { params: Promise<{ name: string }> }) {
  try {
    const { name } = await params;

    // Templates folder is at the root of ai-toolkit (parent of ui folder)
    const templatesPath = path.resolve(process.cwd(), '..', 'templates');
    const templateFile = path.join(templatesPath, `${name}_template.yaml`);

    // if file doesn't exist, return 404
    if (!fs.existsSync(templateFile)) {
      return NextResponse.json({ error: 'Template not found' }, { status: 404 });
    }

    const content = fs.readFileSync(templateFile, 'utf-8');
    return NextResponse.json({ name, content });
  } catch (error) {
    console.error('Error reading template:', error);
    return NextResponse.json({ error: 'Failed to fetch template' }, { status: 500 });
  }
}

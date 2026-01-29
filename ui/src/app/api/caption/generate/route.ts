import { NextRequest, NextResponse } from 'next/server';
import { generateCaption, getServiceStatus, startService } from '@/utils/captionService';

export async function POST(request: NextRequest) {
  const body = await request.json();
  const { imagePath, prompt, maxTokens } = body;

  if (!imagePath) {
    return NextResponse.json(
      { error: 'imagePath is required' },
      { status: 400 }
    );
  }

  // Check if service is running, start if needed
  const status = await getServiceStatus();
  if (!status.running) {
    const startResult = await startService();
    if (!startResult.success) {
      return NextResponse.json(
        { error: `Failed to start caption service: ${startResult.error}` },
        { status: 500 }
      );
    }
  }

  // Generate caption
  const result = await generateCaption(imagePath, { prompt, maxTokens });

  if (result.error) {
    return NextResponse.json(
      { error: result.error, modelState: result.modelState },
      { status: 500 }
    );
  }

  return NextResponse.json({
    caption: result.caption,
    modelState: result.modelState,
  });
}

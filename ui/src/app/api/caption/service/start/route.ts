import { NextResponse } from 'next/server';
import { startService, getServiceStatus } from '@/utils/captionService';

export async function POST() {
  console.log('[API] caption/service/start called');

  const result = await startService();
  console.log('[API] startService result:', result);

  if (result.success) {
    // Get current status after starting
    const status = await getServiceStatus();
    console.log('[API] Service status:', status);
    return NextResponse.json({
      success: true,
      modelState: status.modelState,
    });
  }

  console.error('[API] Failed to start caption service:', result.error);
  return NextResponse.json(
    { success: false, error: result.error },
    { status: 500 }
  );
}

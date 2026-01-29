import { NextResponse } from 'next/server';
import { stopService } from '@/utils/captionService';

export async function POST() {
  const result = await stopService();

  if (result.success) {
    return NextResponse.json({ success: true, modelState: 'stopped' });
  }

  return NextResponse.json(
    { success: false, error: result.error },
    { status: 500 }
  );
}

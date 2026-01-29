import { NextResponse } from 'next/server';
import { getServiceStatus } from '@/utils/captionService';

export async function GET() {
  const status = await getServiceStatus();
  return NextResponse.json(status);
}

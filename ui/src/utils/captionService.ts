/**
 * Caption service management utilities.
 *
 * The caption service is a Python Flask app that runs on localhost:5111
 * and provides on-demand image captioning using QwenVL.
 */

import { spawn } from 'child_process';
import path from 'path';
import fs from 'fs';

const CAPTION_SERVICE_PORT = 5111;
const CAPTION_SERVICE_URL = `http://127.0.0.1:${CAPTION_SERVICE_PORT}`;
const IDLE_TIMEOUT_SECONDS = 600; // 10 minutes

// Get toolkit root - process.cwd() is the ui/ directory when running Next.js
const TOOLKIT_ROOT = path.resolve(process.cwd(), '..');
const PID_FILE = path.join(TOOLKIT_ROOT, '.caption_service.pid');
const LOG_FILE = path.join(TOOLKIT_ROOT, 'output', 'caption_service.log');

// Debug logging for path resolution
console.log('[CaptionService] TOOLKIT_ROOT:', TOOLKIT_ROOT);
console.log('[CaptionService] Script path:', path.join(TOOLKIT_ROOT, 'scripts', 'caption_service.py'));

export type ServiceState = 'stopped' | 'cold' | 'loading' | 'ready' | 'error';

export interface ServiceStatus {
  running: boolean;
  modelState: ServiceState;
  error?: string;
}

/**
 * Check if the caption service is running and get its status.
 */
export async function getServiceStatus(): Promise<ServiceStatus> {
  try {
    const response = await fetch(`${CAPTION_SERVICE_URL}/status`, {
      method: 'GET',
      signal: AbortSignal.timeout(2000),
    });

    if (response.ok) {
      const data = await response.json();
      return {
        running: true,
        modelState: data.model_state as ServiceState,
      };
    }

    return { running: false, modelState: 'stopped' };
  } catch (error) {
    // Service not responding
    return { running: false, modelState: 'stopped' };
  }
}

/**
 * Start the caption service if not already running.
 */
export async function startService(): Promise<{ success: boolean; error?: string }> {
  // Check if already running
  const status = await getServiceStatus();
  if (status.running) {
    return { success: true };
  }

  // Find Python path (same logic as startJob.ts)
  const isWindows = process.platform === 'win32';
  let pythonPath = 'python';
  let useUv = false;

  if (fs.existsSync(path.join(TOOLKIT_ROOT, 'pyproject.toml'))) {
    const homeDir = process.env.HOME || process.env.USERPROFILE || '';
    const uvCandidates = [
      path.join(homeDir, '.local', 'bin', 'uv'),
      path.join(homeDir, '.cargo', 'bin', 'uv'),
      '/usr/local/bin/uv',
      'uv',
    ];
    pythonPath = uvCandidates.find(p => p === 'uv' || fs.existsSync(p)) || 'uv';
    useUv = true;
  } else if (fs.existsSync(path.join(TOOLKIT_ROOT, '.venv'))) {
    pythonPath = isWindows
      ? path.join(TOOLKIT_ROOT, '.venv', 'Scripts', 'python.exe')
      : path.join(TOOLKIT_ROOT, '.venv', 'bin', 'python');
  } else if (fs.existsSync(path.join(TOOLKIT_ROOT, 'venv'))) {
    pythonPath = isWindows
      ? path.join(TOOLKIT_ROOT, 'venv', 'Scripts', 'python.exe')
      : path.join(TOOLKIT_ROOT, 'venv', 'bin', 'python');
  }

  const scriptPath = path.join(TOOLKIT_ROOT, 'scripts', 'caption_service.py');
  console.log('[CaptionService] Looking for script at:', scriptPath);
  console.log('[CaptionService] Python path:', pythonPath);
  console.log('[CaptionService] Using uv:', useUv);

  if (!fs.existsSync(scriptPath)) {
    console.error('[CaptionService] Script not found at:', scriptPath);
    return { success: false, error: `caption_service.py not found at ${scriptPath}` };
  }

  // Ensure log directory exists
  const logDir = path.dirname(LOG_FILE);
  if (!fs.existsSync(logDir)) {
    fs.mkdirSync(logDir, { recursive: true });
  }

  try {
    const args = useUv
      ? ['run', 'python', scriptPath, '--port', String(CAPTION_SERVICE_PORT), '--timeout', String(IDLE_TIMEOUT_SECONDS)]
      : [scriptPath, '--port', String(CAPTION_SERVICE_PORT), '--timeout', String(IDLE_TIMEOUT_SECONDS)];

    console.log('[CaptionService] Spawning:', pythonPath, args.join(' '));
    console.log('[CaptionService] Log file:', LOG_FILE);

    const logFd = fs.openSync(LOG_FILE, 'a');

    const subprocess = spawn(pythonPath, args, {
      detached: true,
      stdio: ['ignore', logFd, logFd],
      env: {
        ...process.env,
        PYTHONUNBUFFERED: '1',
      },
      cwd: TOOLKIT_ROOT,
    });

    // Log spawn errors
    subprocess.on('error', (err) => {
      console.error('[CaptionService] Spawn error:', err);
    });

    // Write PID file
    if (subprocess.pid) {
      console.log('[CaptionService] Process started with PID:', subprocess.pid);
      fs.writeFileSync(PID_FILE, String(subprocess.pid));
    } else {
      console.error('[CaptionService] No PID returned from spawn');
    }

    // Detach from parent process
    subprocess.unref();

    // Wait a moment for the service to start
    await new Promise(resolve => setTimeout(resolve, 2000));

    // Verify it started
    const newStatus = await getServiceStatus();
    console.log('[CaptionService] Status after start:', newStatus);

    if (newStatus.running) {
      return { success: true };
    }

    // Check if there's anything in the log file
    try {
      const logContent = fs.readFileSync(LOG_FILE, 'utf-8').slice(-500);
      console.error('[CaptionService] Log tail:', logContent);
      return { success: false, error: `Service failed to start. Log: ${logContent}` };
    } catch {
      return { success: false, error: 'Service failed to start - no log output' };
    }
  } catch (error: any) {
    console.error('[CaptionService] Error starting service:', error);
    return { success: false, error: error?.message || 'Unknown error' };
  }
}

/**
 * Stop the caption service.
 */
export async function stopService(): Promise<{ success: boolean; error?: string }> {
  try {
    // Try graceful shutdown first
    const response = await fetch(`${CAPTION_SERVICE_URL}/shutdown`, {
      method: 'POST',
      signal: AbortSignal.timeout(5000),
    });

    if (response.ok) {
      // Clean up PID file
      if (fs.existsSync(PID_FILE)) {
        fs.unlinkSync(PID_FILE);
      }
      return { success: true };
    }
  } catch (error) {
    // Service might already be stopped or unresponsive
  }

  // Force kill using PID file
  if (fs.existsSync(PID_FILE)) {
    try {
      const pid = parseInt(fs.readFileSync(PID_FILE, 'utf-8').trim(), 10);
      if (!isNaN(pid)) {
        process.kill(pid, 'SIGTERM');
      }
      fs.unlinkSync(PID_FILE);
    } catch (error) {
      // Process might already be dead
    }
  }

  return { success: true };
}

/**
 * Generate a caption for an image.
 */
export async function generateCaption(
  imagePath: string,
  options?: {
    prompt?: string;
    maxTokens?: number;
  }
): Promise<{ caption?: string; error?: string; modelState?: ServiceState }> {
  try {
    const response = await fetch(`${CAPTION_SERVICE_URL}/caption`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        image_path: imagePath,
        prompt: options?.prompt,
        max_tokens: options?.maxTokens,
      }),
      // Long timeout for first request (model loading)
      signal: AbortSignal.timeout(120000),
    });

    const data = await response.json();

    if (response.ok) {
      return {
        caption: data.caption,
        modelState: data.model_state,
      };
    }

    return { error: data.error || 'Unknown error' };
  } catch (error: any) {
    if (error.name === 'TimeoutError') {
      return { error: 'Request timed out - model may still be loading' };
    }
    return { error: error?.message || 'Service unavailable' };
  }
}

export const CAPTION_SERVICE_PORT_NUMBER = CAPTION_SERVICE_PORT;

import sharp from 'sharp';
import fs from 'fs';
import path from 'path';

const HASH_SIZE = 8;
const RESIZE_SIZE = 32;
const CACHE_VERSION = '1.0.0';
const CACHE_FILENAME = '.aitk_phash.json';

interface CacheEntry {
  hash: string;
  signature: string;
}

interface CacheFile {
  __version__: string;
  [relativePath: string]: string | CacheEntry;
}

/**
 * Compute 1D DCT-II (unnormalized) of a row of length N.
 */
function dct1d(input: Float64Array): Float64Array {
  const N = input.length;
  const output = new Float64Array(N);
  for (let k = 0; k < N; k++) {
    let sum = 0;
    for (let n = 0; n < N; n++) {
      sum += input[n] * Math.cos((Math.PI * (2 * n + 1) * k) / (2 * N));
    }
    output[k] = sum;
  }
  return output;
}

/**
 * Apply separable 2D DCT-II: DCT on rows, then DCT on columns.
 */
function dct2d(matrix: Float64Array[], size: number): Float64Array[] {
  // DCT on rows
  const rowTransformed: Float64Array[] = [];
  for (let i = 0; i < size; i++) {
    rowTransformed.push(dct1d(matrix[i]));
  }

  // DCT on columns
  const result: Float64Array[] = Array.from({ length: size }, () => new Float64Array(size));
  const col = new Float64Array(size);
  for (let j = 0; j < size; j++) {
    for (let i = 0; i < size; i++) {
      col[i] = rowTransformed[i][j];
    }
    const transformed = dct1d(col);
    for (let i = 0; i < size; i++) {
      result[i][j] = transformed[i];
    }
  }

  return result;
}

/**
 * Compute a perceptual hash for an image file.
 * Returns a 16-character hex string (64-bit hash).
 */
export async function computeImagePHash(imagePath: string): Promise<string> {
  // Resize to 32x32 grayscale
  const pixelData = await sharp(imagePath)
    .resize(RESIZE_SIZE, RESIZE_SIZE, { fit: 'fill' })
    .grayscale()
    .raw()
    .toBuffer();

  // Build matrix from pixel data
  const matrix: Float64Array[] = [];
  for (let i = 0; i < RESIZE_SIZE; i++) {
    const row = new Float64Array(RESIZE_SIZE);
    for (let j = 0; j < RESIZE_SIZE; j++) {
      row[j] = pixelData[i * RESIZE_SIZE + j];
    }
    matrix.push(row);
  }

  // Apply 2D DCT
  const dctResult = dct2d(matrix, RESIZE_SIZE);

  // Extract top-left 8x8 low-frequency block (skip [0][0] DC component)
  const lowFreq: number[] = [];
  for (let i = 0; i < HASH_SIZE; i++) {
    for (let j = 0; j < HASH_SIZE; j++) {
      if (i === 0 && j === 0) continue;
      lowFreq.push(dctResult[i][j]);
    }
  }

  // Compute median
  const sorted = [...lowFreq].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  const median = sorted.length % 2 === 0 ? (sorted[mid - 1] + sorted[mid]) / 2 : sorted[mid];

  // Build 64-bit hash: each bit = 1 if value > median
  const bits: number[] = [];
  for (let i = 0; i < HASH_SIZE; i++) {
    for (let j = 0; j < HASH_SIZE; j++) {
      bits.push(dctResult[i][j] > median ? 1 : 0);
    }
  }

  // Convert 64 bits to 16-char hex string
  let hex = '';
  for (let i = 0; i < 64; i += 4) {
    const nibble = (bits[i] << 3) | (bits[i + 1] << 2) | (bits[i + 2] << 1) | bits[i + 3];
    hex += nibble.toString(16);
  }

  return hex;
}

function getFileSignature(filePath: string): string {
  const stat = fs.statSync(filePath);
  return `${stat.size}:${Math.floor(stat.mtimeMs / 1000)}`;
}

function loadCache(cachePath: string): CacheFile {
  try {
    if (fs.existsSync(cachePath)) {
      const data = JSON.parse(fs.readFileSync(cachePath, 'utf-8'));
      if (data.__version__ === CACHE_VERSION) {
        return data;
      }
    }
  } catch (e) {
    console.warn('pHash cache corrupted, will recompute:', e);
  }
  return { __version__: CACHE_VERSION };
}

function saveCache(cachePath: string, cache: CacheFile): void {
  try {
    fs.writeFileSync(cachePath, JSON.stringify(cache, null, 2), 'utf-8');
  } catch (e) {
    console.warn('Failed to write pHash cache:', e);
  }
}

/**
 * Get or compute pHash values for a list of image paths within a dataset folder.
 * Uses a .aitk_phash.json sidecar cache file for persistence.
 */
export async function getOrComputeHashes(
  datasetFolder: string,
  imagePaths: string[]
): Promise<Record<string, string>> {
  const cachePath = path.join(datasetFolder, CACHE_FILENAME);
  const cache = loadCache(cachePath);
  const result: Record<string, string> = {};
  const toCompute: { abs: string; rel: string }[] = [];

  for (const absPath of imagePaths) {
    const relPath = path.relative(datasetFolder, absPath);
    const signature = getFileSignature(absPath);
    const cached = cache[relPath] as CacheEntry | undefined;

    if (cached && typeof cached === 'object' && cached.signature === signature) {
      result[absPath] = cached.hash;
    } else {
      toCompute.push({ abs: absPath, rel: relPath });
    }
  }

  if (toCompute.length > 0) {
    for (const { abs, rel } of toCompute) {
      try {
        const hash = await computeImagePHash(abs);
        const signature = getFileSignature(abs);
        cache[rel] = { hash, signature };
        result[abs] = hash;
      } catch (e) {
        console.warn(`Failed to compute pHash for ${abs}:`, e);
      }
    }
    saveCache(cachePath, cache);
  }

  return result;
}

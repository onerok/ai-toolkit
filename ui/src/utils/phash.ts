/**
 * Compute Hamming distance between two hex hash strings.
 */
export function hammingDistance(hash1: string, hash2: string): number {
  // Compare hex strings nibble-by-nibble to avoid BigInt literal syntax
  let count = 0;
  const len = Math.min(hash1.length, hash2.length);
  for (let i = 0; i < len; i++) {
    let xor = parseInt(hash1[i], 16) ^ parseInt(hash2[i], 16);
    while (xor > 0) {
      count += xor & 1;
      xor >>= 1;
    }
  }
  return count;
}

/**
 * Sort images by visual similarity using greedy nearest-neighbor traversal.
 * Starts at the first image and always picks the closest unvisited neighbor.
 */
export function sortByPHashSimilarity<T extends { phash?: string }>(images: T[]): T[] {
  const hashable = images.filter(img => img.phash);
  const unhashable = images.filter(img => !img.phash);

  if (hashable.length <= 1) {
    return [...hashable, ...unhashable];
  }

  const visited = new Set<number>();
  const sorted: T[] = [];

  // Start with the first image
  let current = 0;
  visited.add(current);
  sorted.push(hashable[current]);

  while (sorted.length < hashable.length) {
    let bestIdx = -1;
    let bestDist = Infinity;

    for (let i = 0; i < hashable.length; i++) {
      if (visited.has(i)) continue;
      const dist = hammingDistance(hashable[current].phash!, hashable[i].phash!);
      if (dist < bestDist) {
        bestDist = dist;
        bestIdx = i;
      }
    }

    if (bestIdx === -1) break;
    visited.add(bestIdx);
    sorted.push(hashable[bestIdx]);
    current = bestIdx;
  }

  return [...sorted, ...unhashable];
}

#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["pillow"]
# ///
"""
Rename images using Hilbert-curve-encoded perceptual hash for sortable similarity.
Also renames corresponding .json and .txt files if present.

The Hilbert encoding ensures similar images have lexicographically close filenames,
and names are stable — adding new images doesn't require renaming existing ones.

Format: {hilbert_hash}_{width}x{height}.{ext}

Usage:
    ./scripts/rename_by_phash.py /path/to/images
    ./scripts/rename_by_phash.py /path/to/images --dry-run
    ./scripts/rename_by_phash.py /path/to/images -r  # recursive
"""

import argparse
import math
import sys
from pathlib import Path

from PIL import Image

HASH_SIZE = 8
RESIZE_SIZE = 32
IMG_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.gif'}


def dct1d(input_arr: list[float]) -> list[float]:
    """Compute 1D DCT-II (unnormalized)."""
    N = len(input_arr)
    output = []
    for k in range(N):
        total = 0.0
        for n in range(N):
            total += input_arr[n] * math.cos((math.pi * (2 * n + 1) * k) / (2 * N))
        output.append(total)
    return output


def dct2d(matrix: list[list[float]], size: int) -> list[list[float]]:
    """Apply separable 2D DCT-II: DCT on rows, then DCT on columns."""
    row_transformed = [dct1d(row) for row in matrix]

    result = [[0.0] * size for _ in range(size)]
    for j in range(size):
        col = [row_transformed[i][j] for i in range(size)]
        transformed = dct1d(col)
        for i in range(size):
            result[i][j] = transformed[i]

    return result


def xy_to_hilbert(n: int, x: int, y: int) -> int:
    """
    Convert (x, y) coordinates to Hilbert curve index.
    n must be a power of 2 (grid size n x n).
    """
    d = 0
    s = n // 2
    while s > 0:
        rx = 1 if (x & s) > 0 else 0
        ry = 1 if (y & s) > 0 else 0
        d += s * s * ((3 * rx) ^ ry)
        # Rotate quadrant
        if ry == 0:
            if rx == 1:
                x = s - 1 - x
                y = s - 1 - y
            x, y = y, x
        s //= 2
    return d


def hilbert_to_xy(n: int, d: int) -> tuple[int, int]:
    """
    Convert Hilbert curve index to (x, y) coordinates.
    n must be a power of 2 (grid size n x n).
    """
    x = y = 0
    s = 1
    while s < n:
        rx = 1 & (d // 2)
        ry = 1 & (d ^ rx)
        # Rotate quadrant
        if ry == 0:
            if rx == 1:
                x = s - 1 - x
                y = s - 1 - y
            x, y = y, x
        x += s * rx
        y += s * ry
        d //= 4
        s *= 2
    return x, y


def compute_phash_hilbert(image_path: str) -> str:
    """
    Compute a perceptual hash with Hilbert curve encoding.
    Returns a 16-character hex string that sorts by visual similarity.
    """
    with Image.open(image_path) as img:
        img = img.convert('L').resize((RESIZE_SIZE, RESIZE_SIZE), Image.LANCZOS)
        pixels = list(img.getdata())

    matrix = []
    for i in range(RESIZE_SIZE):
        row = [float(pixels[i * RESIZE_SIZE + j]) for j in range(RESIZE_SIZE)]
        matrix.append(row)

    dct_result = dct2d(matrix, RESIZE_SIZE)

    # Extract top-left 8x8 low-frequency block (skip [0][0] DC component)
    low_freq = []
    for i in range(HASH_SIZE):
        for j in range(HASH_SIZE):
            if i == 0 and j == 0:
                continue
            low_freq.append(dct_result[i][j])

    sorted_freq = sorted(low_freq)
    mid = len(sorted_freq) // 2
    if len(sorted_freq) % 2 == 0:
        median = (sorted_freq[mid - 1] + sorted_freq[mid]) / 2
    else:
        median = sorted_freq[mid]

    # Build 8x8 bit grid
    bit_grid = [[0] * HASH_SIZE for _ in range(HASH_SIZE)]
    for i in range(HASH_SIZE):
        for j in range(HASH_SIZE):
            bit_grid[i][j] = 1 if dct_result[i][j] > median else 0

    # Collect bits in Hilbert curve order
    bits = []
    for d in range(HASH_SIZE * HASH_SIZE):
        x, y = hilbert_to_xy(HASH_SIZE, d)
        bits.append(bit_grid[y][x])

    # Convert 64 bits to 16-char hex string
    hex_str = ''
    for i in range(0, 64, 4):
        nibble = (bits[i] << 3) | (bits[i + 1] << 2) | (bits[i + 2] << 1) | bits[i + 3]
        hex_str += format(nibble, 'x')

    return hex_str


def get_image_info(image_path: Path) -> tuple[str, int, int] | None:
    """Get phash, width, and height of an image. Returns None on error."""
    try:
        phash = compute_phash_hilbert(str(image_path))
        with Image.open(image_path) as img:
            width, height = img.size
        return phash, width, height
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None


def rename_files(
    image_path: Path,
    new_stem: str,
    dry_run: bool = False
) -> bool:
    """
    Rename an image and its sidecar files (.json, .txt).
    Returns True if renamed, False if skipped.
    """
    new_name = f"{new_stem}{image_path.suffix.lower()}"
    new_path = image_path.parent / new_name

    # Skip if already named correctly
    if image_path.name == new_name:
        print(f"  Skip (already named): {image_path.name}")
        return False

    # Check for conflicts
    if new_path.exists() and new_path != image_path:
        print(f"  Skip (conflict): {image_path.name} -> {new_name} (target exists)")
        return False

    # Collect files to rename
    old_stem = image_path.stem
    sidecar_exts = ['.json', '.txt']
    files_to_rename = [(image_path, new_path)]

    for ext in sidecar_exts:
        sidecar = image_path.parent / f"{old_stem}{ext}"
        if sidecar.exists():
            new_sidecar = image_path.parent / f"{new_stem}{ext}"
            if new_sidecar.exists() and new_sidecar != sidecar:
                print(f"  Skip (sidecar conflict): {sidecar.name} -> {new_sidecar.name}")
                return False
            files_to_rename.append((sidecar, new_sidecar))

    # Perform renames
    for old, new in files_to_rename:
        if dry_run:
            print(f"  Would rename: {old.name} -> {new.name}")
        else:
            old.rename(new)
            print(f"  Renamed: {old.name} -> {new.name}")

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Rename images using Hilbert-encoded pHash for sortable similarity"
    )
    parser.add_argument("directory", help="Directory containing images")
    parser.add_argument(
        "--dry-run", "-n", action="store_true",
        help="Show what would be renamed without doing it"
    )
    parser.add_argument(
        "--recursive", "-r", action="store_true",
        help="Process subdirectories recursively"
    )
    args = parser.parse_args()

    directory = Path(args.directory)
    if not directory.is_dir():
        print(f"Error: {directory} is not a directory")
        sys.exit(1)

    if args.dry_run:
        print("DRY RUN - no files will be renamed\n")

    # Find all images
    if args.recursive:
        images = [p for p in directory.rglob("*") if p.suffix.lower() in IMG_EXTENSIONS]
    else:
        images = [p for p in directory.iterdir() if p.suffix.lower() in IMG_EXTENSIONS]

    images.sort()
    print(f"Found {len(images)} images in {directory}\n")

    # Compute info and rename
    print("Computing Hilbert-encoded perceptual hashes...")
    renamed = 0
    processed = 0

    for image_path in images:
        info = get_image_info(image_path)
        if not info:
            continue
        processed += 1

        phash, width, height = info
        new_stem = f"{phash}_{width}x{height}"

        print(f"Processing: {image_path.name}")
        if rename_files(image_path, new_stem, args.dry_run):
            renamed += 1

    print(f"\nProcessed: {processed}/{len(images)} images")
    print(f"{'Would rename' if args.dry_run else 'Renamed'}: {renamed}/{processed} images")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Download a small subset of naruto-blip-captions for performance benchmarking.

Usage:
    uv run python scripts/download_perf_dataset.py [--count 20] [--output datasets/perf_test]
"""

import argparse
import json
import sys
from pathlib import Path
from urllib.request import urlopen, Request
from urllib.error import HTTPError

DATASET_API = "https://datasets-server.huggingface.co/first-rows"
DATASET_NAME = "lambdalabs/naruto-blip-captions"


def download_dataset(output_dir: Path, count: int = 20) -> None:
    """Download images and captions from HuggingFace datasets API."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check if already populated
    existing = list(output_dir.glob("*.txt"))
    if len(existing) >= count:
        print(f"Dataset already exists with {len(existing)} images at {output_dir}")
        return

    print(f"Downloading {count} images from {DATASET_NAME}...")

    # Fetch rows from HuggingFace datasets API (first-rows endpoint returns ~100 rows)
    url = f"{DATASET_API}?dataset={DATASET_NAME}&config=default&split=train"

    try:
        req = Request(url, headers={"User-Agent": "ai-toolkit-perf-benchmark"})
        with urlopen(req, timeout=30) as response:
            data = json.loads(response.read().decode())
    except HTTPError as e:
        print(f"Error fetching dataset metadata: {e}")
        sys.exit(1)

    rows = data.get("rows", [])
    if not rows:
        print("No rows returned from API")
        sys.exit(1)

    # Limit to requested count
    rows = rows[:count]
    print(f"Fetched metadata for {len(rows)} images")

    downloaded = 0
    for i, row_entry in enumerate(rows):
        row = row_entry.get("row", {})
        image_info = row.get("image", {})
        caption = row.get("text", "a naruto character")

        # Get image URL
        image_url = image_info.get("src")
        if not image_url:
            print(f"  [{i+1}/{len(rows)}] Skipping: no image URL")
            continue

        image_path = output_dir / f"{i:04d}.jpg"
        caption_path = output_dir / f"{i:04d}.txt"

        if image_path.exists() and caption_path.exists():
            print(f"  [{i+1}/{len(rows)}] Already exists: {image_path.name}")
            downloaded += 1
            continue

        try:
            req = Request(image_url, headers={"User-Agent": "ai-toolkit-perf-benchmark"})
            with urlopen(req, timeout=30) as img_response:
                image_data = img_response.read()

            image_path.write_bytes(image_data)
            caption_path.write_text(caption)
            print(f"  [{i+1}/{len(rows)}] Downloaded: {image_path.name} ({len(image_data)//1024}KB)")
            downloaded += 1
        except Exception as e:
            print(f"  [{i+1}/{len(rows)}] Failed: {e}")
            continue

    print(f"\nDone! {downloaded} images saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Download perf test dataset")
    parser.add_argument("--count", type=int, default=20, help="Number of images to download")
    parser.add_argument("--output", type=Path, default=Path("datasets/perf_test"), help="Output directory")
    args = parser.parse_args()

    download_dataset(args.output, args.count)


if __name__ == "__main__":
    main()

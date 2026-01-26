#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "filetype",
# ]
# ///
"""
Find image files where the file extension doesn't match the actual image type.

Usage:
    uv run check_image_extensions.py [directory] [--fix] [--verbose]

Examples:
    uv run check_image_extensions.py .
    uv run check_image_extensions.py ~/Pictures --verbose
    uv run check_image_extensions.py ./images --fix
"""

import argparse
import filetype
from pathlib import Path

# Mapping of MIME types to their canonical extensions
MIME_TO_EXTENSIONS: dict[str, set[str]] = {
    "image/jpeg": {".jpg", ".jpeg", ".jpe", ".jfif"},
    "image/png": {".png"},
    "image/gif": {".gif"},
    "image/webp": {".webp"},
    "image/bmp": {".bmp", ".dib"},
    "image/tiff": {".tiff", ".tif"},
    "image/x-icon": {".ico"},
    "image/vnd.microsoft.icon": {".ico"},
    "image/heic": {".heic", ".heics"},
    "image/heif": {".heif", ".heifs"},
    "image/avif": {".avif"},
    "image/svg+xml": {".svg"},
    "image/x-tga": {".tga"},
    "image/vnd.adobe.photoshop": {".psd"},
    "image/x-canon-cr2": {".cr2"},
    "image/x-nikon-nef": {".nef"},
    "image/x-sony-arw": {".arw"},
    "image/x-panasonic-rw2": {".rw2"},
    "image/x-fuji-raf": {".raf"},
    "image/x-olympus-orf": {".orf"},
    "image/jxl": {".jxl"},
}

# Preferred extension for each MIME type (used with --fix)
MIME_TO_PREFERRED: dict[str, str] = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/gif": ".gif",
    "image/webp": ".webp",
    "image/bmp": ".bmp",
    "image/tiff": ".tiff",
    "image/x-icon": ".ico",
    "image/vnd.microsoft.icon": ".ico",
    "image/heic": ".heic",
    "image/heif": ".heif",
    "image/avif": ".avif",
    "image/svg+xml": ".svg",
    "image/x-tga": ".tga",
    "image/vnd.adobe.photoshop": ".psd",
    "image/x-canon-cr2": ".cr2",
    "image/x-nikon-nef": ".nef",
    "image/x-sony-arw": ".arw",
    "image/x-panasonic-rw2": ".rw2",
    "image/x-fuji-raf": ".raf",
    "image/x-olympus-orf": ".orf",
    "image/jxl": ".jxl",
}

# Common image extensions to check (files without these are skipped unless they're detected as images)
IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".jpe", ".jfif", ".png", ".gif", ".webp", ".bmp", ".dib",
    ".tiff", ".tif", ".ico", ".heic", ".heics", ".heif", ".heifs", ".avif",
    ".svg", ".tga", ".psd", ".cr2", ".nef", ".arw", ".rw2", ".raf", ".orf",
    ".jxl", ".raw", ".img",
}


def check_file(filepath: Path, verbose: bool = False) -> tuple[bool, str | None, str | None]:
    """
    Check if a file's extension matches its actual image type.
    
    Returns:
        (is_mismatch, detected_mime, message)
    """
    try:
        kind = filetype.guess(filepath)
        ext = filepath.suffix.lower()
        
        # File is not a recognized image type
        if kind is None:
            if ext in IMAGE_EXTENSIONS:
                return True, None, f"Has image extension '{ext}' but content is not a recognized image"
            return False, None, None
        
        # Not an image file
        if not kind.mime.startswith("image/"):
            if ext in IMAGE_EXTENSIONS:
                return True, kind.mime, f"Has image extension '{ext}' but is actually '{kind.mime}'"
            return False, None, None
        
        # Check if extension matches the detected type
        valid_extensions = MIME_TO_EXTENSIONS.get(kind.mime, set())
        
        if not valid_extensions:
            if verbose:
                print(f"  [INFO] Unknown image MIME type: {kind.mime} for {filepath}")
            return False, kind.mime, None
        
        if ext not in valid_extensions:
            expected = ", ".join(sorted(valid_extensions))
            return True, kind.mime, f"Extension '{ext}' doesn't match detected type '{kind.mime}' (expected: {expected})"
        
        return False, kind.mime, None
        
    except PermissionError:
        if verbose:
            print(f"  [SKIP] Permission denied: {filepath}")
        return False, None, None
    except Exception as e:
        if verbose:
            print(f"  [ERROR] Could not read {filepath}: {e}")
        return False, None, None


def fix_extension(filepath: Path, detected_mime: str, dry_run: bool = False) -> Path | None:
    """Rename file to have the correct extension."""
    preferred_ext = MIME_TO_PREFERRED.get(detected_mime)
    if not preferred_ext:
        return None
    
    new_path = filepath.with_suffix(preferred_ext)
    
    # Handle collision
    if new_path.exists() and new_path != filepath:
        counter = 1
        while new_path.exists():
            new_path = filepath.with_stem(f"{filepath.stem}_{counter}").with_suffix(preferred_ext)
            counter += 1
    
    if not dry_run:
        filepath.rename(new_path)
    
    return new_path


def scan_directory(
    directory: Path,
    fix: bool = False,
    verbose: bool = False,
    dry_run: bool = False,
) -> list[dict]:
    """Recursively scan directory for image extension mismatches."""
    mismatches = []
    scanned = 0
    
    for filepath in directory.rglob("*"):
        if not filepath.is_file():
            continue
        
        # Skip hidden files and common non-image files
        if filepath.name.startswith("."):
            continue
        
        scanned += 1
        if verbose and scanned % 1000 == 0:
            print(f"  Scanned {scanned} files...")
        
        is_mismatch, detected_mime, message = check_file(filepath, verbose)
        
        if is_mismatch:
            result = {
                "path": filepath,
                "message": message,
                "detected_mime": detected_mime,
                "fixed": False,
                "new_path": None,
            }
            
            if fix and detected_mime:
                new_path = fix_extension(filepath, detected_mime, dry_run)
                if new_path:
                    result["fixed"] = True
                    result["new_path"] = new_path
            
            mismatches.append(result)
    
    if verbose:
        print(f"  Total files scanned: {scanned}")
    
    return mismatches


def main():
    parser = argparse.ArgumentParser(
        description="Find image files where the extension doesn't match the actual image type.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s .                     # Scan current directory
  %(prog)s ~/Pictures -v         # Scan with verbose output
  %(prog)s ./images --fix        # Fix mismatched extensions
  %(prog)s ./images --fix --dry-run  # Preview fixes without renaming
        """,
    )
    parser.add_argument(
        "directory",
        nargs="?",
        default=".",
        help="Directory to scan (default: current directory)",
    )
    parser.add_argument(
        "-f", "--fix",
        action="store_true",
        help="Rename files to have the correct extension",
    )
    parser.add_argument(
        "-n", "--dry-run",
        action="store_true",
        help="Show what would be renamed without actually renaming (use with --fix)",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Show verbose output including skipped files and progress",
    )
    
    args = parser.parse_args()
    
    directory = Path(args.directory).resolve()
    
    if not directory.exists():
        print(f"Error: Directory '{directory}' does not exist")
        return 1
    
    if not directory.is_dir():
        print(f"Error: '{directory}' is not a directory")
        return 1
    
    print(f"Scanning: {directory}")
    if args.fix:
        if args.dry_run:
            print("Mode: Dry run (showing what would be fixed)")
        else:
            print("Mode: Fix mismatched extensions")
    print()
    
    mismatches = scan_directory(
        directory,
        fix=args.fix,
        verbose=args.verbose,
        dry_run=args.dry_run,
    )
    
    if not mismatches:
        print("✓ No mismatched image extensions found!")
        return 0
    
    print(f"Found {len(mismatches)} file(s) with mismatched extensions:\n")
    
    for item in mismatches:
        filepath = item["path"]
        rel_path = filepath.relative_to(directory) if filepath.is_relative_to(directory) else filepath
        
        print(f"  {rel_path}")
        print(f"    └─ {item['message']}")
        
        if item["fixed"]:
            new_rel = item["new_path"].relative_to(directory) if item["new_path"].is_relative_to(directory) else item["new_path"]
            action = "Would rename" if args.dry_run else "Renamed"
            print(f"    └─ {action} to: {new_rel}")
        
        print()
    
    if args.fix and not args.dry_run:
        fixed_count = sum(1 for m in mismatches if m["fixed"])
        print(f"Fixed {fixed_count}/{len(mismatches)} files")
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
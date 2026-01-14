#!/usr/bin/env python3
"""
Extract images from Water Bodies Dataset based on a list file.

This script reads a text file containing image names (without extensions),
finds the corresponding images in the source directory, and copies them
to the output directory.

Usage:
    python wbs_extract.py <list_file> <source_dir> <output_dir>

Example:
    python wbs_extract.py wbs-si_val.txt /path/to/source /path/to/output
"""


import os
import shutil
import argparse
from pathlib import Path


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Extract images from Water Bodies Dataset based on a list file'
    )
    parser.add_argument(
        'list_file',
        type=str,
        help='Path to the text file containing image names (without extensions)'
    )
    parser.add_argument(
        'source_dir',
        type=str,
        help='Path to the source directory containing the images'
    )
    parser.add_argument(
        'output_dir',
        type=str,
        help='Path to the output directory where images will be copied'
    )
    return parser.parse_args()


def main():
    """Main function to extract and copy images."""
    args = parse_args()
    
    # Convert to Path objects
    list_file = Path(args.list_file)
    source_dir = Path(args.source_dir)
    output_dir = Path(args.output_dir)
    
    # Validate input paths
    if not list_file.exists():
        print(f"Error: List file not found: {list_file}")
        return
    
    if not source_dir.exists():
        print(f"Error: Source directory not found: {source_dir}")
        return
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Read image names from list file
    with open(list_file, 'r') as f:
        image_names = [line.strip() for line in f if line.strip()]
    
    print(f"Found {len(image_names)} image names in {list_file}")
    
    # Copy images
    copied_count = 0
    not_found_count = 0
    
    for image_name in image_names:
        # Try to find the image with .jpg extension
        image_file = source_dir / f"{image_name}.jpg"
        
        if image_file.exists():
            # Copy the image to output directory
            shutil.copy2(image_file, output_dir / image_file.name)
            copied_count += 1
        else:
            print(f"Warning: Image not found: {image_file}")
            not_found_count += 1
    
    print(f"\nSummary:")
    print(f"  Total images in list: {len(image_names)}")
    print(f"  Successfully copied: {copied_count}")
    print(f"  Not found: {not_found_count}")
    print(f"  Output directory: {output_dir}")


if __name__ == '__main__':
    main()

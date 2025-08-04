#!/usr/bin/env python3
"""
Data Preparation Script for Histopathology Images

This script helps organize your histology images into proper train/validation splits
and checks data quality requirements.
"""

import os
import shutil
import random
from pathlib import Path
from PIL import Image
import numpy as np
import argparse


def prepare_histology_data(source_dir, output_dir, train_split=0.8, copy_files=True):
    """
    Organize histology images into train/val splits
    
    Args:
        source_dir: Directory containing all your histology images
        output_dir: Output directory (will create train/val subdirs)
        train_split: Fraction for training (0.8 = 80% train, 20% val)
        copy_files: If True, copy files; if False, create symlinks
    """
    print("Histopathology Data Preparation")
    print("=" * 50)
    
    # Create output directories
    train_dir = Path(output_dir) / "train"
    val_dir = Path(output_dir) / "val"
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all image files
    image_extensions = {'.png', '.jpg', '.jpeg', '.tiff', '.tif', '.PNG', '.JPG', '.JPEG', '.TIFF', '.TIF'}
    image_files = [f for f in Path(source_dir).rglob('*') 
                   if f.suffix in image_extensions and f.is_file()]
    
    print(f"Found {len(image_files)} images in {source_dir}")
    
    if len(image_files) == 0:
        print("No images found! Check your source directory and file extensions.")
        return
    
    # Filter by size (minimum 256x256)
    valid_files = []
    for img_path in image_files:
        try:
            with Image.open(img_path) as img:
                if img.size[0] >= 256 and img.size[1] >= 256:
                    valid_files.append(img_path)
                else:
                    print(f"  Skipping {img_path.name}: too small ({img.size[0]}x{img.size[1]})")
        except Exception as e:
            print(f"Error reading {img_path.name}: {e}")
    
    print(f"{len(valid_files)} images meet size requirements (≥256×256)")
    
    if len(valid_files) < 10:
        print("Warning: Very few images found. Consider collecting more data.")
    
    # Shuffle and split
    random.seed(42)  # For reproducibility
    random.shuffle(valid_files)
    split_idx = int(len(valid_files) * train_split)
    
    train_files = valid_files[:split_idx]
    val_files = valid_files[split_idx:]
    
    # Copy or link files
    action = "Copying" if copy_files else "Linking"
    print(f"\n{action} files to training/validation sets...")
    
    def transfer_file(src, dst, copy_files):
        if copy_files:
            shutil.copy2(src, dst)
        else:
            os.symlink(src.absolute(), dst)
    
    # Process training files
    print(f"{action} {len(train_files)} images to training set...")
    for file in train_files:
        dst = train_dir / file.name
        if not dst.exists():
            transfer_file(file, dst, copy_files)
    
    # Process validation files
    print(f"{action} {len(val_files)} images to validation set...")
    for file in val_files:
        dst = val_dir / file.name
        if not dst.exists():
            transfer_file(file, dst, copy_files)
    
    print("\nData preparation complete!")
    print(f"Training images: {len(train_files)}")
    print(f"Validation images: {len(val_files)}")
    print(f"Train/Val split: {train_split:.1%}/{1-train_split:.1%}")


def analyze_dataset(data_dir, sample_size=100):
    """Analyze your histology dataset for quality and characteristics"""

    print("\nDataset Analysis")
    print("=" * 50)
    
    # Find all image files
    image_files = []
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.tif')):
                image_files.append(os.path.join(root, file))
    
    print(f"Total images found: {len(image_files)}")
    
    if len(image_files) == 0:
        print("No images found in the specified directory!")
        return
    
    # Sample images for analysis
    sample_files = random.sample(image_files, min(sample_size, len(image_files)))
    print(f"Analyzing {len(sample_files)} sample images...")
    
    # Analyze image properties
    sizes = []
    formats = {}
    file_sizes = []
    errors = []
    
    for img_path in sample_files:
        try:
            with Image.open(img_path) as img:
                sizes.append(img.size)
                fmt = img.format or 'Unknown'
                formats[fmt] = formats.get(fmt, 0) + 1
                
                # File size in MB
                file_size = os.path.getsize(img_path) / (1024 * 1024)
                file_sizes.append(file_size)
                
        except Exception as e:
            errors.append(f"{os.path.basename(img_path)}: {str(e)}")
    
    # Print statistics
    if sizes:
        widths, heights = zip(*sizes)
        print(f"\nImage Dimensions:")
        print(f"   Width range: {min(widths):,} - {max(widths):,} pixels")
        print(f"   Height range: {min(heights):,} - {max(heights):,} pixels")
        print(f"   Average size: {np.mean(widths):.0f} × {np.mean(heights):.0f}")
        print(f"   Most common size: {max(set(sizes), key=sizes.count)}")
    
    if file_sizes:
        print(f"\nFile Sizes:")
        print(f"   Range: {min(file_sizes):.2f} - {max(file_sizes):.2f} MB")
        print(f"   Average: {np.mean(file_sizes):.2f} MB")
        print(f"   Total estimated: {np.mean(file_sizes) * len(image_files):.1f} MB")
    
    print(f"\nFile Formats:")
    for fmt, count in sorted(formats.items()):
        percentage = (count / len(sample_files)) * 100
        print(f"   {fmt}: {count} files ({percentage:.1f}%)")
    
    # Quality checks
    print(f"\nQuality Assessment:")
    
    # Size requirements
    small_images = sum(1 for w, h in sizes if w < 256 or h < 256)
    if small_images > 0:
        print(f"   !!! {small_images}/{len(sizes)} images are smaller than 256×256")
    else:
        print("   All sampled images meet minimum size requirements")
    
    # Aspect ratio check
    aspect_ratios = [w/h for w, h in sizes]
    extreme_ratios = sum(1 for ar in aspect_ratios if ar < 0.5 or ar > 2.0)
    if extreme_ratios > 0:
        print(f"   {extreme_ratios}/{len(sizes)} images have extreme aspect ratios")
    else:
        print("   Good aspect ratio distribution")
    
    # File size check
    large_files = sum(1 for fs in file_sizes if fs > 50)  # > 50 MB
    if large_files > 0:
        print(f"   {large_files}/{len(file_sizes)} files are very large (>50MB)")
    
    # Error report
    if errors:
        print(f"\nErrors encountered:")
        for error in errors[:5]:  # Show first 5 errors
            print(f"   {error}")
        if len(errors) > 5:
            print(f"   ... and {len(errors) - 5} more errors")
    else:
        print("   No errors reading sample images")
    
    # Recommendations
    print(f"\nRecommendations:")
    
    if len(image_files) < 500:
        print("   Consider collecting more images (500+ recommended for basic training)")
    
    if np.mean([w for w, h in sizes]) < 512:
        print("   Consider using 256×256 patch size for training")
    else:
        print("   You can use 512×512 patch size for better quality")
    
    if small_images > len(sizes) * 0.1:  # More than 10% are too small
        print("   Consider filtering out small images or cropping larger patches")
    
    total_size_gb = (np.mean(file_sizes) * len(image_files)) / 1024
    if total_size_gb > 50:
        print(f"   Large dataset ({total_size_gb:.1f}GB) - ensure adequate storage")


def main():
    parser = argparse.ArgumentParser(description="Prepare histopathology data for training")
    parser.add_argument("source_dir", help="Directory containing your histology images")
    parser.add_argument("--output_dir", default="data", help="Output directory for organized data")
    parser.add_argument("--train_split", type=float, default=0.8, help="Fraction for training (default: 0.8)")
    parser.add_argument("--analyze_only", action="store_true", help="Only analyze data, don't organize")
    parser.add_argument("--symlink", action="store_true", help="Create symlinks instead of copying files")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.source_dir):
        print(f"Source directory '{args.source_dir}' does not exist!")
        return
    
    if args.analyze_only:
        analyze_dataset(args.source_dir)
    else:
        prepare_histology_data(
            source_dir=args.source_dir,
            output_dir=args.output_dir,
            train_split=args.train_split,
            copy_files=not args.symlink
        )
        
        # Also run analysis on the prepared data
        if os.path.exists(args.output_dir):
            analyze_dataset(args.output_dir)


if __name__ == "__main__":
    # Example usage if run without arguments
    if len(os.sys.argv) == 1:
        print("Histopathology Data Preparation Tool")
        print("=" * 50)
        print("\nUsage examples:")
        print("  python prepare_data.py /path/to/your/images")
        print("  python prepare_data.py /path/to/your/images --output_dir my_data")
        print("  python prepare_data.py /path/to/your/images --train_split 0.9")
        print("  python prepare_data.py /path/to/your/images --analyze_only")
        print("  python prepare_data.py /path/to/your/images --symlink")
        print("\nArguments:")
        print("  source_dir     : Directory containing your histology images")
        print("  --output_dir   : Output directory (default: 'data')")
        print("  --train_split  : Training fraction (default: 0.8)")
        print("  --analyze_only : Only analyze, don't organize files")
        print("  --symlink      : Create symlinks instead of copying")
        print("\nThe script will:")
        print("  [TRUE] Find all image files in source directory")
        print("  [TRUE] Filter images by size (≥256×256)")
        print("  [TRUE] Split into train/validation sets")
        print("  [TRUE] Analyze dataset characteristics")
        print("  [TRUE] Provide training recommendations")
    else:
        main()

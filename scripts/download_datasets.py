#!/usr/bin/env python3
"""
Easy Dataset Downloader for Histopathology Images

This script provides easy access to public histopathology datasets
for training the autoencoder models.
"""

import os
import sys
import subprocess
import requests
import zipfile
from pathlib import Path
import argparse


def install_kaggle():
    """Install Kaggle API if not present."""
    try:
        import kaggle
        return True
    except ImportError:
        print("Installing Kaggle API...")
        subprocess.run([sys.executable, "-m", "pip", "install", "kaggle"])
        return True


def setup_kaggle_credentials():
    """Help user set up Kaggle credentials."""
    kaggle_dir = Path.home() / ".kaggle"
    kaggle_file = kaggle_dir / "kaggle.json"
    
    if kaggle_file.exists():
        return True
    
    print("\nKaggle API Setup Required")
    print("=" * 40)
    print("1. Go to https://www.kaggle.com/account")
    print("2. Click 'Create New API Token'")
    print("3. Download kaggle.json file")
    print("4. Place it in:", kaggle_dir)
    print("\nOr run these commands:")
    print(f"mkdir -p {kaggle_dir}")
    print(f"cp /path/to/downloaded/kaggle.json {kaggle_file}")
    print(f"chmod 600 {kaggle_file}")
    
    input("\nPress Enter when you've set up the credentials...")
    return kaggle_file.exists()


def download_breast_cancer_dataset(output_dir="downloaded_data"):
    """Download breast cancer histopathology dataset from Kaggle."""
    print("\nDownloading Breast Cancer Histopathology Dataset")
    print("=" * 60)
    print("Dataset: Breast Histopathology Images")
    print("Size: ~1.3GB (277,524 patches)")
    print("Format: PNG (50x50 pixels)")
    print("Classes: Invasive Ductal Carcinoma (IDC)")
    
    if not install_kaggle():
        return False
    
    if not setup_kaggle_credentials():
        print(" Kaggle credentials not set up properly.")
        return False
    
    try:
        import kaggle
        
        # Download dataset
        print("\n Downloading dataset...")
        dataset_name = "paultimothymooney/breast-histopathology-images"
        download_path = Path(output_dir) / "breast_cancer_raw"
        download_path.mkdir(parents=True, exist_ok=True)
        
        kaggle.api.dataset_download_files(
            dataset_name, 
            path=str(download_path), 
            unzip=True
        )
        
        print("Download complete!")
        return str(download_path)
        
    except Exception as e:
        print(f"Download failed: {e}")
        return False


def download_patchcamelyon_dataset(output_dir="downloaded_data"):
    """Download PatchCamelyon dataset from Kaggle."""
    print("\nDownloading PatchCamelyon Dataset")
    print("=" * 60)
    print("Dataset: PatchCamelyon (PCam)")
    print("Size: ~7.2GB (327,680 patches)")
    print("Format: TIF (96x96 pixels)")
    print("Classes: Metastatic tissue detection")
    
    if not install_kaggle():
        return False
    
    if not setup_kaggle_credentials():
        print("Kaggle credentials not set up properly.")
        return False
    
    try:
        import kaggle
        
        # Download dataset
        print("\n Downloading dataset...")
        dataset_name = "jejjohnson/patchcamelyon"
        download_path = Path(output_dir) / "patchcamelyon_raw"  
        download_path.mkdir(parents=True, exist_ok=True)
        
        kaggle.api.dataset_download_files(
            dataset_name,
            path=str(download_path),
            unzip=True
        )
        
        print(" Download complete!")  
        return str(download_path)
        
    except Exception as e:
        print(f" Download failed: {e}")
        return False


def download_colorectal_dataset(output_dir="downloaded_data"):
    """Download colorectal histology dataset."""
    print("\n Downloading Colorectal Cancer Dataset")
    print("=" * 60)
    print("Dataset: Colorectal Histology MNIST")
    print("Size: ~150MB (5,000 images)")
    print("Format: TIF (64x64 pixels)")
    print("Classes: 8 tissue types")
    
    if not install_kaggle():
        return False
    
    if not setup_kaggle_credentials():
        print(" Kaggle credentials not set up properly.")
        return False
    
    try:
        import kaggle
        
        # Download dataset
        print("\n Downloading dataset...")
        dataset_name = "kmader/colorectal-histology-mnist"
        download_path = Path(output_dir) / "colorectal_raw"
        download_path.mkdir(parents=True, exist_ok=True)
        
        kaggle.api.dataset_download_files(
            dataset_name,
            path=str(download_path), 
            unzip=True
        )
        
        print(" Download complete!")
        return str(download_path)
        
    except Exception as e:
        print(f" Download failed: {e}")
        return False


def download_sample_openslide_data(output_dir="downloaded_data"):
    """Download sample data from OpenSlide."""
    print("\n Downloading OpenSlide Sample Data")
    print("=" * 60)
    print("Dataset: OpenSlide Demo Images")
    print("Size: ~50MB (sample WSI files)")
    print("Format: Various WSI formats")
    print("Classes: Demo/test images")
    
    download_path = Path(output_dir) / "openslide_samples"
    download_path.mkdir(parents=True, exist_ok=True)
    
    # Sample URLs (these are example URLs - replace with actual OpenSlide demo URLs)
    sample_urls = [
        "https://openslide.cs.cmu.edu/download/openslide-testdata/Aperio/CMU-1-Small-Region.svs",
        "https://openslide.cs.cmu.edu/download/openslide-testdata/Aperio/CMU-1.svs"
    ]
    
    try:
        for i, url in enumerate(sample_urls):
            print(f" Downloading sample {i+1}/{len(sample_urls)}...")
            response = requests.get(url)
            if response.status_code == 200:
                filename = download_path / f"sample_{i+1}.svs"
                with open(filename, 'wb') as f:
                    f.write(response.content)
                print(f" Downloaded: {filename}")
            else:
                print(f" Could not download sample {i+1}")
        
        return str(download_path)
        
    except Exception as e:
        print(f" Download failed: {e}")
        return False


def prepare_downloaded_data(raw_data_path, output_dir="training_data"):
    """Prepare downloaded data using the project's preparation script."""
    print(f"\n Preparing Data from {raw_data_path}")
    print("=" * 60)
    
    try:
        # Run the data preparation script
        cmd = [
            sys.executable, "-m", "src.training.prepare_data",
            raw_data_path,
            "--output_dir", output_dir
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(" Data preparation complete!")
            print(f" Training data ready in: {output_dir}/")
            print("\n You can now start training with:")
            print(f"   python -m src.training.train_real_data {output_dir}/train")
            return True
        else:
            print(" Data preparation failed:")
            print(result.stderr)
            return False
            
    except Exception as e:
        print(f" Data preparation error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Download histopathology datasets")
    parser.add_argument("--dataset", choices=[
        "breast-cancer", "patchcamelyon", "colorectal", "openslide", "all"
    ], default="breast-cancer", help="Dataset to download")
    parser.add_argument("--output_dir", default="downloaded_data", 
                       help="Output directory for downloads")
    parser.add_argument("--prepare", action="store_true", 
                       help="Automatically prepare data after download")
    
    args = parser.parse_args()
    
    print(" Histopathology Dataset Downloader")
    print("=" * 50)
    print("This tool helps you download public histopathology datasets")
    print("for training the autoencoder models.")
    print("=" * 50)
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    downloaded_paths = []
    
    # Download selected datasets
    if args.dataset == "breast-cancer" or args.dataset == "all":
        path = download_breast_cancer_dataset(args.output_dir)
        if path:
            downloaded_paths.append(path)
    
    if args.dataset == "patchcamelyon" or args.dataset == "all":
        path = download_patchcamelyon_dataset(args.output_dir)
        if path:
            downloaded_paths.append(path)
    
    if args.dataset == "colorectal" or args.dataset == "all":
        path = download_colorectal_dataset(args.output_dir)
        if path:
            downloaded_paths.append(path)
    
    if args.dataset == "openslide" or args.dataset == "all":
        path = download_sample_openslide_data(args.output_dir)
        if path:
            downloaded_paths.append(path)
    
    # Prepare data if requested
    if args.prepare and downloaded_paths:
        print(f"\n Preparing {len(downloaded_paths)} dataset(s)...")
        for i, path in enumerate(downloaded_paths):
            output_name = f"prepared_data_{i+1}"
            prepare_downloaded_data(path, output_name)
    
    # Show summary
    print("\n" + "=" * 60)
    print(" DOWNLOAD SUMMARY")
    print("=" * 60)
    
    if downloaded_paths:
        print(f" Successfully downloaded {len(downloaded_paths)} dataset(s):")
        for path in downloaded_paths:
            print(f"    {path}")
        
        if not args.prepare:
            print("\nNext steps:")
            print("1. Activate environment: source slide_env/bin/activate")
            print("2. Prepare data: python -m src.training.prepare_data [dataset_path]")
            print("3. Start training: python -m src.training.train_real_data data/train")
    else:
        print(" No datasets were downloaded successfully.")
        print("Please check your internet connection and Kaggle credentials.")
    
    print("=" * 60)


if __name__ == "__main__":
    main()

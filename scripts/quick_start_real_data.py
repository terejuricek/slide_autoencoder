#!/usr/bin/env python3
"""
Quick Start Guide for Real Data Training

This script provides step-by-step instructions and checks for training
autoencoder models on real histopathology data.
"""

import os
import sys
from pathlib import Path


def check_dependencies():
    """Check if required dependencies are available"""
    
    print("Checking Dependencies")
    print("=" * 40)
    
    dependencies = {
        'torch': 'PyTorch',
        'PIL': 'Pillow', 
        'numpy': 'NumPy',
        'matplotlib.pyplot': 'Matplotlib'
    }
    
    optional_dependencies = {
        'cv2': 'OpenCV (pip install opencv-python)',
        'torchvision': 'TorchVision'
    }
    
    missing = []
    missing_optional = []
    
    # Check required dependencies
    for module, name in dependencies.items():
        try:
            __import__(module)
            print(f"[FOUND] {name}")
        except ImportError as e:
            print(f"[MISSING] {name} - REQUIRED")
            if "matplotlib" in module:
                missing.append("pip install matplotlib")
            else:
                missing.append(f"pip install {module}")
        except Exception as e:
            print(f"[ERROR] {name} - {str(e)[:50]}...")
            if "matplotlib" in module:
                missing.append("pip install matplotlib")
            else:
                missing.append(f"pip install {module}")
    
    # Check optional dependencies
    for module, name in optional_dependencies.items():
        try:
            __import__(module)
            print(f"[FOUND] {name}")
        except ImportError:
            print(f"[OPTIONAL] {name} - recommended")
            missing_optional.append(name.split('(')[1].replace(')', '') if '(' in name else f"pip install {module}")
    
    if missing:
        print(f"\nMissing required dependencies:")
        for cmd in missing:
            print(f"   {cmd}")
        return False
    
    if missing_optional:
        print(f"\nInstall optional dependencies for full functionality:")
        for cmd in missing_optional:
            print(f"   {cmd}")
    
    print(f"\nDependencies check complete!")
    return True


def print_image_requirements():
    """Print detailed image requirements""" 
    
    print("\nImage Requirements for Training")
    print("=" * 50)
    
    print("""
SUPPORTED FORMATS:
   - PNG (recommended - lossless)
   - JPEG/JPG (acceptable - check quality)
   - TIFF/TIF (excellent - medical standard)

SIZE REQUIREMENTS:
   Minimum: 256x256 pixels
   Recommended: 512x512 pixels or larger
   Maximum: No limit (will be patched automatically)

COLOR REQUIREMENTS:
   - RGB images (3 channels)
   - 8-bit per channel (minimum)
   - 16-bit per channel (preferred for medical images)

DATASET SIZE:
   Minimum for testing: 50-100 images
   Good results: 500+ images
   Best results: 1000+ images
   Split: 80% training, 20% validation

QUALITY GUIDELINES:
   - Well-focused images
   - Proper staining (H&E, IHC, etc.)
   - Minimal compression artifacts
   - Consistent lighting
   - Clear tissue structures

AVOID:
   - Severely out of focus images
   - Over/under-stained samples
   - Heavy compression artifacts
   - Extreme lighting variations
   - Tissue folds or tears
""")


def print_project_structure():
    """Check and display project structure"""
    
    print("\nProject Structure Validation")
    print("=" * 40)
    
    required_files = [
        "src/slide_autoencoder/models.py",
        "src/training/train_real_data.py", 
        "src/training/prepare_data.py",
        "src/evaluation/evaluate_model.py"
    ]
    
    all_present = True
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"[FOUND] {file_path}")
        else:
            print(f"[MISSING] {file_path}")
            all_present = False
    
    if all_present:
        print("\nProject structure validation: PASSED")
    else:
        print("\nProject structure validation: FAILED")
        print("Please ensure you're running this from the project root directory.")
    
    return all_present


def print_training_guide():
    """Print step-by-step training instructions"""
    
    print("\nStep-by-Step Training Guide")
    print("=" * 40)
    
    print("""
STEP 1: Prepare Your Data
   python -m src.training.prepare_data /path/to/your/images

STEP 2: Choose Your Model
   Basic Model (34.5M parameters):
   - Faster training (2-4 hours)
   - Lower GPU memory (~2GB)
   - Good for most cases
   
   Deep Model (133.3M parameters):
   - Better quality but slower (6-10 hours)
   - Higher GPU memory (~4GB)
   - Best for research/high-quality needs

STEP 3: Start Training
   Basic model:
   python -m src.training.train_real_data data/train --model_type basic
   
   Deep model:
   python -m src.training.train_real_data data/train --model_type deep

STEP 4: Monitor Training
   - Training progress will be displayed
   - Best model automatically saved
   - Training plots generated

STEP 5: Evaluate Results  
   python -m src.evaluation.evaluate_model checkpoints/best_model.pth data/val
""")


def print_hardware_recommendations():
    """Print hardware-specific recommendations"""
    
    print("\nHardware Recommendations")
    print("=" * 30)
    
    print("""
GPU MEMORY RECOMMENDATIONS:
   4GB GPU:  Basic model, batch_size=4, patch_size=256
   8GB GPU:  Basic model, batch_size=8, patch_size=256
   12GB+ GPU: Deep model, batch_size=8, patch_size=512

CPU-ONLY TRAINING:
   Possible but very slow (10-100x slower)
   Use: python -m src.training.train_real_data data/train --device cpu

TRAINING TIME ESTIMATES:
   Basic model (1000 images): 2-4 hours on GPU
   Deep model (1000 images): 6-10 hours on GPU
""")


def print_troubleshooting():
    """Print common troubleshooting tips"""
    
    print("\nTroubleshooting Common Issues")
    print("=" * 35)
    
    print("""
MEMORY ERRORS:
   - Reduce batch_size: --batch_size 4 (or 2)
   - Reduce patch_size: --patch_size 224
   - Use basic model instead of deep
   - Close other GPU applications

POOR RESULTS:
   - Ensure good image quality
   - Increase training epochs: --num_epochs 200
   - Try different learning rate: --learning_rate 0.0001
   - Collect more training data

TRAINING CRASHES:
   - Check image files for corruption
   - Ensure sufficient disk space
   - Verify stable power supply
   - Monitor system temperature

IMPORT ERRORS:
   - Install package: pip install -e .
   - Check Python path includes project root
   - Verify all dependencies installed
""")


def print_quick_commands():
    """Print quick command reference"""
    
    print("\nQuick Command Reference")
    print("=" * 25)
    
    print("""
# Install package
pip install -e .

# Prepare data
python -m src.training.prepare_data /path/to/images

# Quick training (basic model)
python -m src.training.train_real_data data/train

# Advanced training (deep model)
python -m src.training.train_real_data data/train --model_type deep --batch_size 4

# Evaluate model
python -m src.evaluation.evaluate_model checkpoints/best_model.pth data/val

# View model architecture
python -m src.visualization.model_visualization
""")


def main():
    """Main function to run the quick start guide"""
    
    print("Slide Autoencoder - Quick Start Guide")
    print("=" * 50)
    
    # Check dependencies
    if not check_dependencies():
        print("\nPlease install missing dependencies before proceeding.")
        return
    
    # Check project structure
    if not print_project_structure():
        return
    
    # Print all guidance
    print_image_requirements()
    print_training_guide()
    print_hardware_recommendations()
    print_troubleshooting()
    print_quick_commands()
    
    print("\n" + "=" * 50)
    print("Quick Start Guide Complete!")
    print("Follow the steps above to train your autoencoder model.")
    print("=" * 50)


if __name__ == "__main__":
    main()

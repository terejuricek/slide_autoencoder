#!/usr/bin/env python3
"""
Simple demonstration script for histopathology autoencoder
This script shows how to train and use the autoencoder in practice
"""

import os
import sys

def install_dependencies():
    """Install required dependencies"""
    print("Installing dependencies...")
    os.system("pip install torch torchvision numpy matplotlib pillow opencv-python tqdm scikit-learn")
    print("Dependencies installed!")

def create_demo_data():
    """Create synthetic demonstration data"""
    print("Creating synthetic histology data...")
    
    try:
        from data_utils import create_synthetic_patches
        
        # Create directories
        os.makedirs("data/train", exist_ok=True)
        os.makedirs("data/test", exist_ok=True)
        
        # Create training data
        create_synthetic_patches("data/train", num_patches=100, patch_size=256)
        print("Created 100 training patches")
        
        # Create test data
        create_synthetic_patches("data/test", num_patches=20, patch_size=256)
        print("Created 20 test patches")
        
        print("Demo data created successfully!")
        return True
        
    except ImportError as e:
        print(f"Error creating demo data: {e}")
        print("Please install dependencies first")
        return False

def train_demo_model():
    """Train a demonstration model"""
    print("Training demonstration model...")
    
    try:
        from train import train_autoencoder
        
        # Train basic autoencoder with small dataset
        trainer, history = train_autoencoder(
            data_dir="data/train",
            model_type="basic",
            batch_size=4,        # Small batch size for demo
            num_epochs=10,       # Few epochs for demo
            learning_rate=1e-3,  # Higher learning rate for faster convergence
            patch_size=256
        )
        
        print("Model training completed!")
        print("Saved model to: checkpoints/best_model.pth")
        return True
        
    except Exception as e:
        print(f"Error during training: {e}")
        return False

def test_inference():
    """Test inference on demo data"""
    print("Testing inference...")
    
    try:
        from inference import HistologyDenoiser
        
        # Load trained model
        denoiser = HistologyDenoiser("checkpoints/best_model.pth", model_type="basic")
        
        # Process test images
        denoiser.batch_denoise_directory("data/test", "results/denoised")
        
        print("Inference completed!")
        print("Results saved to: results/denoised/")
        return True
        
    except Exception as e:
        print(f"Error during inference: {e}")
        return False

def show_results():
    """Show some results"""
    print("Showing results...")
    
    try:
        import matplotlib.pyplot as plt
        import cv2
        import numpy as np
        import os
        
        # Find some result files
        test_dir = "data/test"
        results_dir = "results/denoised"
        
        if not os.path.exists(results_dir):
            print("No results found")
            return False
        
        # Get first available image
        test_files = [f for f in os.listdir(test_dir) if f.endswith('.png')]
        result_files = [f for f in os.listdir(results_dir) if f.endswith('.png')]
        
        if not test_files or not result_files:
            print("No image files found")
            return False
        
        # Load and display first image pair
        test_img = cv2.imread(os.path.join(test_dir, test_files[0]))
        result_img = cv2.imread(os.path.join(results_dir, result_files[0]))
        
        if test_img is None or result_img is None:
            print("Could not load images")
            return False
        
        # Convert BGR to RGB
        test_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)
        result_img = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
        
        # Show comparison
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        
        axes[0].imshow(test_img)
        axes[0].set_title('Original (with synthetic noise)')
        axes[0].axis('off')
        
        axes[1].imshow(result_img)
        axes[1].set_title('Denoised by Autoencoder')
        axes[1].axis('off')
        
        plt.tight_layout()
        plt.savefig("demo_results.png", dpi=150, bbox_inches='tight')
        plt.show()
        
        print("Results visualization saved as demo_results.png")
        return True
        
    except Exception as e:
        print(f"Error showing results: {e}")
        return False

def main():
    """Main demonstration function"""
    print("=" * 60)
    print("HISTOPATHOLOGY AUTOENCODER DEMONSTRATION")
    print("=" * 60)
    
    # Check if dependencies are available
    try:
        import torch
        print("✓ PyTorch is available")
    except ImportError:
        print("✗ PyTorch not found. Installing dependencies...")
        install_dependencies()
    
    # Step 1: Create demo data
    print("\nStep 1: Creating demonstration data...")
    if not create_demo_data():
        print("Failed to create demo data. Exiting.")
        return
    
    # Step 2: Train model
    print("\nStep 2: Training autoencoder model...")
    if not train_demo_model():
        print("Failed to train model. Exiting.")
        return
    
    # Step 3: Test inference
    print("\nStep 3: Testing inference...")
    if not test_inference():
        print("Failed to run inference. Exiting.")
        return
    
    # Step 4: Show results
    print("\nStep 4: Displaying results...")
    show_results()
    
    print("\n" + "=" * 60)
    print("DEMONSTRATION COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    
    print("""
Next steps for real usage:

1. Replace synthetic data with real histology images:
   - Place your .png, .jpg, .tiff files in data/train/
   - Optionally create data/val/ for validation

2. Adjust training parameters:
   - Increase num_epochs (50-200 for real training)
   - Adjust batch_size based on your GPU memory
   - Use validation data for better training

3. Fine-tune model architecture:
   - Try "deep" model for better quality
   - Adjust base_channels for model size
   - Modify noise_factor in data_utils.py

4. Optimize for your specific images:
   - Adjust patch_size based on your image resolution
   - Modify data augmentation parameters
   - Tune loss function weights

5. Production deployment:
   - Use inference.py for batch processing
   - Implement proper error handling
   - Add logging and monitoring
    """)

if __name__ == "__main__":
    main()

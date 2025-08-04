#!/usr/bin/env python3
"""
Simple Training Script for Real Histopathology Data

This script provides an easy-to-use interface for training autoencoder models
on real histopathology images with minimal configuration required.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import argparse
from pathlib import Path
import json
from datetime import datetime

from src.slide_autoencoder.models import HistoAutoencoder, DeepHistoAutoencoder
from data_utils import get_data_loaders
from train import HistologyTrainer


def train_with_real_data(
    train_dir,
    val_dir=None,
    model_type="basic",
    patch_size=256,
    batch_size=8,
    num_epochs=100,
    learning_rate=1e-4,
    output_dir="training_output",
    device="auto"
):
    """
    Train autoencoder model with real histopathology data
    
    Args:
        train_dir: Directory containing training images
        val_dir: Directory containing validation images (optional)
        model_type: "basic" or "deep"
        patch_size: Size of patches to use (256 or 512)
        batch_size: Batch size for training
        num_epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        output_dir: Directory to save outputs
        device: Device to use ("auto", "cuda", "cpu")
    
    Returns:
        Dictionary with training results
    """
    
    print("Training Autoencoder on Real Histopathology Data")
    print("=" * 60)
    
    # Setup device
    if device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
    
    print(f"Using device: {device}")
    print(f"Model type: {model_type}")
    print(f"Patch size: {patch_size}×{patch_size}")
    print(f"Batch size: {batch_size}")
    print(f"Epochs: {num_epochs}")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save training configuration
    config = {
        'train_dir': train_dir,
        'val_dir': val_dir,
        'model_type': model_type,
        'patch_size': patch_size,
        'batch_size': batch_size,
        'num_epochs': num_epochs,
        'learning_rate': learning_rate,
        'device': str(device),
        'start_time': datetime.now().isoformat()
    }
    
    with open(Path(output_dir) / "training_config.json", 'w') as f:
        json.dump(config, f, indent=2)
    
    # Check data directories
    if not os.path.exists(train_dir):
        raise ValueError(f"Training directory '{train_dir}' does not exist!")
    
    train_images = list(Path(train_dir).glob('*.png')) + list(Path(train_dir).glob('*.jpg')) + \
                   list(Path(train_dir).glob('*.jpeg')) + list(Path(train_dir).glob('*.tiff')) + \
                   list(Path(train_dir).glob('*.tif'))

    print(f"Found {len(train_images)} training images")

    if len(train_images) == 0:
        raise ValueError("No training images found! Check your data directory.")
    
    if len(train_images) < 50:
        print("Warning: Very few training images. Consider collecting more data.")
    
    # Setup data loaders
    print("Setting up data loaders...")

    try:
        train_loader, val_loader = get_data_loaders(
            train_dir=train_dir,
            val_dir=val_dir,
            batch_size=batch_size,
            patch_size=patch_size,
            num_workers=2,  # Conservative for stability
            noise_factor=0.2  # Moderate noise for real data
        )

        print(f"[TRUE] Training batches: {len(train_loader)}")
        if val_loader:
            print(f"[TRUE] Validation batches: {len(val_loader)}")
        else:
            print("No validation set provided")
            
    except Exception as e:
        print(f"Error setting up data loaders: {e}")
        print("Try reducing batch_size or patch_size")
        raise
    
    # Create model
    print(f"Creating {model_type} autoencoder...")

    if model_type == "basic":
        model = HistoAutoencoder(input_channels=3, base_channels=64)
    elif model_type == "deep":
        model = DeepHistoAutoencoder(input_channels=3, base_channels=64)
    else:
        raise ValueError("model_type must be 'basic' or 'deep'")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")
    print(f"Estimated model size: {total_params * 4 / (1024**2):.1f} MB")
    
    # Estimate memory requirements
    estimated_memory = estimate_memory_usage(batch_size, patch_size, total_params)
    print(f"Estimated GPU memory needed: {estimated_memory:.1f} GB")

    if device.type == "cuda":
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"Available GPU memory: {gpu_memory:.1f} GB")

        if estimated_memory > gpu_memory * 0.8:  # 80% threshold
            print("Warning: Estimated memory usage is high. Consider reducing batch_size.")

    # Initialize trainer
    print("Initializing trainer...")
    trainer = HistologyTrainer(
        model=model,
        device=device,
        learning_rate=learning_rate,
        weight_decay=1e-5
    )
    
    # Start training
    print("\nStarting training...")
    print("-" * 60)
    
    try:
        history = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=num_epochs,
            save_dir=output_dir,
            save_frequency=max(1, num_epochs // 10)  # Save 10 checkpoints
        )

        print("\n [TRUE] Training completed successfully!")

        # Save final results
        results = {
            'config': config,
            'final_train_loss': history['train_losses'][-1] if history['train_losses'] else None,
            'final_val_loss': history['val_losses'][-1] if history['val_losses'] else None,
            'best_val_loss': trainer.best_val_loss,
            'total_epochs': len(history['train_losses']),
            'end_time': datetime.now().isoformat(),
            'model_parameters': total_params,
            'device_used': str(device)
        }
        
        with open(Path(output_dir) / "training_results.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        # Print summary
        print_training_summary(results, output_dir)
        
        return results
        
    except Exception as e:
        print(f"\n Training failed: {e}")
        print("Troubleshooting tips:")
        print("   - Reduce batch_size if out of memory")
        print("   - Check image quality and formats")
        print("   - Ensure adequate disk space")
        print("   - Try reducing patch_size")
        raise


def estimate_memory_usage(batch_size, patch_size, model_params):
    """Estimate GPU memory usage in GB"""
    
    # Model parameters (float32)
    model_memory = model_params * 4 / (1024**3)
    
    # Activations (rough estimate)
    activation_memory = batch_size * 3 * patch_size * patch_size * 10 * 4 / (1024**3)  # ~10 activation maps
    
    # Gradients
    gradient_memory = model_memory
    
    # Optimizer states (Adam keeps 2 momentum buffers)
    optimizer_memory = model_memory * 2
    
    # Total with some buffer
    total_memory = (model_memory + activation_memory + gradient_memory + optimizer_memory) * 1.2
    
    return total_memory


def print_training_summary(results, output_dir):
    """Print training summary"""

    print("\nTraining Summary")
    print("=" * 40)
    
    config = results['config']
    
    print(f"Model: {config['model_type']} autoencoder")
    print(f"Training images: {len(list(Path(config['train_dir']).glob('*.*')))}")
    print(f"Patch size: {config['patch_size']}×{config['patch_size']}")
    print(f"Batch size: {config['batch_size']}")
    print(f"Epochs completed: {results.get('total_epochs', 'Unknown')}")
    
    if results.get('final_train_loss'):
        print(f"Final training loss: {results['final_train_loss']:.6f}")
    
    if results.get('final_val_loss'):
        print(f"Final validation loss: {results['final_val_loss']:.6f}")
        print(f"Best validation loss: {results['best_val_loss']:.6f}")
    
    print(f"Model parameters: {results['model_parameters']:,}")
    print(f"Device used: {results['device_used']}")
    
    # Start and end times
    if 'start_time' in config and 'end_time' in results:
        start = datetime.fromisoformat(config['start_time'])
        end = datetime.fromisoformat(results['end_time'])
        duration = end - start
        print(f"Training duration: {duration}")

    print(f"\nOutput files:")
    print(f"   Model checkpoint: {output_dir}/best_model.pth")
    print(f"   Training config: {output_dir}/training_config.json")
    print(f"   Training results: {output_dir}/training_results.json")
    print(f"   Training history: {output_dir}/training_history.png")

    print(f"\nNext steps:")
    print(f"   1. Evaluate model: python evaluate_model.py {output_dir}/best_model.pth <test_dir>")
    print(f"   2. Use for inference: python inference.py")
    print(f"   3. Check training plots: {output_dir}/training_history.png")


def main():
    parser = argparse.ArgumentParser(description="Train autoencoder on real histopathology data")
    
    # Required arguments
    parser.add_argument("train_dir", help="Directory containing training images")
    
    # Optional arguments
    parser.add_argument("--val_dir", help="Directory containing validation images")
    parser.add_argument("--model_type", choices=["basic", "deep"], default="basic", 
                       help="Model architecture (default: basic)")
    parser.add_argument("--patch_size", type=int, default=256, 
                       help="Patch size for training (default: 256)")
    parser.add_argument("--batch_size", type=int, default=8, 
                       help="Batch size (default: 8)")
    parser.add_argument("--num_epochs", type=int, default=100, 
                       help="Number of training epochs (default: 100)")
    parser.add_argument("--learning_rate", type=float, default=1e-4, 
                       help="Learning rate (default: 1e-4)")
    parser.add_argument("--output_dir", default="training_output", 
                       help="Output directory (default: training_output)")
    parser.add_argument("--device", default="auto", 
                       help="Device to use: auto, cuda, cpu (default: auto)")
    
    args = parser.parse_args()
    
    # Validate arguments
    if not os.path.exists(args.train_dir):
        print(f"Training directory '{args.train_dir}' does not exist!")
        return
    
    if args.val_dir and not os.path.exists(args.val_dir):
        print(f"Validation directory '{args.val_dir}' does not exist!")
        return
    
    if args.patch_size not in [224, 256, 384, 512]:
        print(f"Warning: Unusual patch size {args.patch_size}. Recommended: 256 or 512")

    try:
        results = train_with_real_data(
            train_dir=args.train_dir,
            val_dir=args.val_dir,
            model_type=args.model_type,
            patch_size=args.patch_size,
            batch_size=args.batch_size,
            num_epochs=args.num_epochs,
            learning_rate=args.learning_rate,
            output_dir=args.output_dir,
            device=args.device
        )

        print(f"\nTraining completed successfully!")
        print(f"All outputs saved in: {args.output_dir}")
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"\nTraining failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    # Example usage if run without arguments
    if len(os.sys.argv) == 1:
        print("Simple Training Script for Real Histopathology Data")
        print("=" * 60)
        print("\nQuick Start:")
        print("   1. Organize your data: python prepare_data.py /path/to/images")
        print("   2. Train model: python train_real_data.py data/train --val_dir data/val")
        print("\nFull Usage:")
        print("   python train_real_data.py <train_dir> [options]")
        print("\nOptions:")
        print("   --val_dir DIR         Validation directory")
        print("   --model_type TYPE     basic or deep (default: basic)")
        print("   --patch_size SIZE     Patch size (default: 256)")
        print("   --batch_size SIZE     Batch size (default: 8)")
        print("   --num_epochs N        Training epochs (default: 100)")
        print("   --output_dir DIR      Output directory (default: training_output)")
        print("\nExamples:")
        print("   # Basic training")
        print("   python train_real_data.py data/train")
        print()
        print("   # With validation set")
        print("   python train_real_data.py data/train --val_dir data/val")
        print()
        print("   # Deep model with longer training")
        print("   python train_real_data.py data/train --model_type deep --num_epochs 200")
        print()
        print("   # Smaller batch for limited GPU memory")
        print("   python train_real_data.py data/train --batch_size 4")
        print("\nPrerequisites:")
        print("   1. Images organized in train/ (and optionally val/) directories")
        print("   2. Images in PNG, JPG, or TIFF format")
        print("   3. Images at least 256×256 pixels")
        print("   4. GPU recommended for reasonable training time")
    else:
        exit(main())

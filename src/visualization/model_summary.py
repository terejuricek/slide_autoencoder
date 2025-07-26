#!/usr/bin/env python3
"""
Model Summary - Quick Reference

A concise summary of both autoencoder models with key metrics and recommendations.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from model_visualization import count_parameters
    from models import HistoAutoencoder, DeepHistoAutoencoder
    
    def print_model_summary():
        """Print a concise summary of both models."""
        
        print("🔬 HISTOPATHOLOGY AUTOENCODER - MODEL SUMMARY")
        print("=" * 70)
        
        # Create models
        basic_model = HistoAutoencoder(input_channels=3, base_channels=64)
        deep_model = DeepHistoAutoencoder(input_channels=3, base_channels=64)
        
        basic_params, _ = count_parameters(basic_model)
        deep_params, _ = count_parameters(deep_model)
        
        print()
        print("📊 MODEL COMPARISON")
        print("-" * 50)
        print(f"{'Metric':<25} {'Basic':<15} {'Deep':<15}")
        print("-" * 50)
        print(f"{'Parameters':<25} {f'{basic_params/1e6:.1f}M':<15} {f'{deep_params/1e6:.1f}M':<15}")
        print(f"{'Model Size':<25} {f'{basic_params*4/(1024**2):.0f}MB':<15} {f'{deep_params*4/(1024**2):.0f}MB':<15}")
        print(f"{'Training Memory*':<25} {'~2.5GB':<15} {'~6.8GB':<15}")
        print(f"{'Inference Memory*':<25} {'~0.4GB':<15} {'~0.8GB':<15}")
        print(f"{'Training Speed':<25} {'Fast':<15} {'Moderate':<15}")
        print(f"{'Quality':<25} {'Good':<15} {'Superior':<15}")
        print("-" * 50)
        print("*Batch size 8, 256x256 images")
        
        print()
        print("🎯 RECOMMENDATIONS")
        print("-" * 50)
        print("Choose BASIC Autoencoder if:")
        print("  • Working with 256x256 histology patches")
        print("  • Limited GPU memory (< 4GB)")
        print("  • Need fast training/inference")
        print("  • Standard denoising requirements")
        print()
        print("Choose DEEP Autoencoder if:")
        print("  • Working with 512x512+ images")
        print("  • Have adequate GPU memory (6GB+)")
        print("  • Need superior quality")
        print("  • Complex tissue patterns/multiple stains")
        
        print()
        print("⚡ QUICK START")
        print("-" * 50)
        print("1. Run visualization demo:")
        print("   python demo_visualization.py")
        print()
        print("2. Interactive exploration:")
        print("   python model_visualization.py")
        print()
        print("3. Complete overview:")
        print("   python test/overview.py")
        print()
        print("4. Start training:")
        print("   python train.py")
        
        print()
        print("📈 ARCHITECTURE HIGHLIGHTS")
        print("-" * 50)
        print("Both models feature:")
        print("  ✓ U-Net architecture with skip connections")
        print("  ✓ Specialized for histopathology images")
        print("  ✓ Robust noise handling capabilities")
        print("  ✓ Patch-based processing for large slides")
        print("  ✓ Preserves diagnostic details")
        
        print()
        print("🔍 TECHNICAL SPECS")
        print("-" * 50)
        print("Input/Output: RGB images (3 channels)")
        print("Patch size: 256x256 (expandable to 512x512)")
        print("Skip connections: 4 levels for detail preservation")
        print("Activation: LeakyReLU + Sigmoid output")
        print("Normalization: Batch normalization throughout")
        print("Loss function: Combined MSE + L1 + Perceptual")
        
        print()
        print("=" * 70)
        print("For detailed analysis, use the visualization tools!")
        print("=" * 70)

except ImportError as e:
    print(f"❌ Could not import required modules: {e}")
    print("Please ensure all dependencies are installed and models.py exists.")

if __name__ == "__main__":
    print_model_summary()

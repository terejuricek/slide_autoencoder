"""
Model Visualization Module for Histopathology Autoencoders

This module provides comprehensive visualization tools for understanding
the architecture, data flow, and parameters of both autoencoder models.
"""

import torch
import torch.nn as nn
from models import HistoAutoencoder, DeepHistoAutoencoder
import numpy as np
from collections import OrderedDict


def count_parameters(model):
    """Count total and trainable parameters in a model."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def get_layer_info(model, input_shape=(1, 3, 256, 256)):
    """
    Get detailed information about each layer including:
    - Layer name and type
    - Input/Output shapes
    - Number of parameters
    - Memory usage
    """
    device = next(model.parameters()).device
    x = torch.randn(input_shape).to(device)
    
    layer_info = []
    hooks = []
    
    def register_hook(name, module):
        def hook(module, input, output):
            # Handle multiple inputs/outputs
            if isinstance(input, (list, tuple)):
                input_shape = [tuple(i.shape) if hasattr(i, 'shape') else str(i) for i in input]
            else:
                input_shape = tuple(input.shape) if hasattr(input, 'shape') else str(input)
            
            if isinstance(output, (list, tuple)):
                output_shape = [tuple(o.shape) if hasattr(o, 'shape') else str(o) for o in output]
            else:
                output_shape = tuple(output.shape) if hasattr(output, 'shape') else str(output)
            
            # Count parameters
            params = sum(p.numel() for p in module.parameters() if p.requires_grad)
            
            # Estimate memory (rough calculation)
            if isinstance(output, torch.Tensor):
                memory_mb = output.numel() * 4 / (1024 * 1024)  # 4 bytes per float32
            else:
                memory_mb = 0
            
            layer_info.append({
                'name': name,
                'type': module.__class__.__name__,
                'input_shape': input_shape,
                'output_shape': output_shape,
                'parameters': params,
                'memory_mb': memory_mb
            })
        
        return hook
    
    # Register hooks for all modules
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Only leaf modules
            hook = register_hook(name, module)
            hooks.append(module.register_forward_hook(hook))
    
    # Forward pass to trigger hooks
    model.eval()
    with torch.no_grad():
        _ = model(x)
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    return layer_info


def print_architecture_map(model_type="basic", input_shape=(1, 3, 256, 256)):
    """
    Print a detailed architecture map for the specified model.
    """
    print("=" * 100)
    print(f"ARCHITECTURE MAP - {model_type.upper()} AUTOENCODER")
    print("=" * 100)
    
    # Create model
    if model_type == "basic":
        model = HistoAutoencoder(input_channels=3, base_channels=64)
    elif model_type == "deep":
        model = DeepHistoAutoencoder(input_channels=3, base_channels=64)
    else:
        raise ValueError("model_type must be 'basic' or 'deep'")
    
    model.eval()
    
    # Get basic model info
    total_params, trainable_params = count_parameters(model)
    
    print(f"Model Type: {model.__class__.__name__}")
    print(f"Input Shape: {input_shape}")
    print(f"Total Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    print(f"Model Size: {total_params * 4 / (1024*1024):.2f} MB")
    print()
    
    # Get layer information
    layer_info = get_layer_info(model, input_shape)
    
    # Print detailed layer information
    print("LAYER-BY-LAYER BREAKDOWN:")
    print("-" * 100)
    print(f"{'Layer Name':<30} {'Type':<20} {'Input Shape':<25} {'Output Shape':<25} {'Params':<10}")
    print("-" * 100)
    
    total_memory = 0
    for info in layer_info:
        input_str = str(info['input_shape'])[:24]
        output_str = str(info['output_shape'])[:24]
        
        print(f"{info['name']:<30} {info['type']:<20} {input_str:<25} {output_str:<25} {info['parameters']:<10}")
        total_memory += info['memory_mb']
    
    print("-" * 100)
    print(f"Estimated Forward Pass Memory: {total_memory:.2f} MB")
    print()


def print_data_flow_map(model_type="basic"):
    """
    Print a visual representation of data flow through the model.
    """
    print("=" * 100)
    print(f"DATA FLOW MAP - {model_type.upper()} AUTOENCODER")
    print("=" * 100)
    
    if model_type == "basic":
        print_basic_data_flow()
    elif model_type == "deep":
        print_deep_data_flow()
    
    print()


def print_basic_data_flow():
    """Print data flow for basic autoencoder."""
    print("""
    INPUT: [B, 3, 256, 256] - RGB Histology Image
         │
         ▼
    ┌─────────────────────────────────────────────────────────────────────────────────┐
    │                              ENCODER PATHWAY                                    │
    └─────────────────────────────────────────────────────────────────────────────────┘
         │
         ▼
    ENC1: [B, 3, 256, 256] → [B, 64, 128, 128]   ┐
         │ Conv3x3 + BN + LeakyReLU + MaxPool     │ SKIP
         ▼                                        │ CONNECTION
    ENC2: [B, 64, 128, 128] → [B, 128, 64, 64]   │     ┐
         │ Conv3x3 + BN + LeakyReLU + MaxPool     │     │ SKIP
         ▼                                        │     │ CONNECTION
    ENC3: [B, 128, 64, 64] → [B, 256, 32, 32]    │     │     ┐
         │ Conv3x3 + BN + LeakyReLU + MaxPool     │     │     │ SKIP
         ▼                                        │     │     │ CONNECTION
    ENC4: [B, 256, 32, 32] → [B, 512, 16, 16]    │     │     │     ┐
         │ Conv3x3 + BN + LeakyReLU + MaxPool     │     │     │     │ SKIP
         ▼                                        │     │     │     │ CONNECTION
    ┌─────────────────────────────────────────────────────────────────────────────────┐
    │                              BOTTLENECK                                        │
    │ [B, 512, 16, 16] → [B, 1024, 16, 16] → [B, 1024, 16, 16]                     │
    │ Conv3x3 + BN + LeakyReLU + Conv3x3 + BN + LeakyReLU                           │
    └─────────────────────────────────────────────────────────────────────────────────┘
         │
         ▼
    ┌─────────────────────────────────────────────────────────────────────────────────┐
    │                              DECODER PATHWAY                                   │
    └─────────────────────────────────────────────────────────────────────────────────┘
         │
         ▼
    DEC4: [B, 1024, 16, 16] → [B, 512, 32, 32] ←─┘     │     │     │
         │ ConvTranspose + BN + LeakyReLU              │     │     │
         │ + Skip Connection [B, 512, 32, 32]          │     │     │
         │ = [B, 1024, 32, 32] → [B, 512, 32, 32]      │     │     │
         ▼                                              │     │     │
    DEC3: [B, 768, 32, 32] → [B, 256, 64, 64] ←────────┘     │     │
         │ ConvTranspose + BN + LeakyReLU                    │     │
         │ + Skip Connection [B, 256, 64, 64]                │     │
         │ = [B, 512, 64, 64] → [B, 256, 64, 64]             │     │
         ▼                                                    │     │
    DEC2: [B, 384, 64, 64] → [B, 128, 128, 128] ←───────────┘     │
         │ ConvTranspose + BN + LeakyReLU                          │
         │ + Skip Connection [B, 128, 128, 128]                    │
         │ = [B, 256, 128, 128] → [B, 128, 128, 128]               │
         ▼                                                          │
    DEC1: [B, 192, 128, 128] → [B, 64, 256, 256] ←───────────────┘
         │ ConvTranspose + BN + LeakyReLU
         │ + Skip Connection [B, 64, 256, 256]
         │ = [B, 128, 256, 256] → [B, 64, 256, 256]
         ▼
    ┌─────────────────────────────────────────────────────────────────────────────────┐
    │                              FINAL LAYER                                       │
    │ [B, 64, 256, 256] → [B, 64, 256, 256] → [B, 3, 256, 256]                     │
    │ Conv3x3 + BN + LeakyReLU + Conv1x1 + Sigmoid                                   │
    └─────────────────────────────────────────────────────────────────────────────────┘
         │
         ▼
    OUTPUT: [B, 3, 256, 256] - Denoised RGB Image
    
    Key:
    B = Batch Size
    Conv3x3 = 3x3 Convolution
    ConvTranspose = Transposed Convolution (Upsampling)
    BN = Batch Normalization
    Skip Connection = Features from encoder concatenated to decoder
    """)


def print_deep_data_flow():
    """Print data flow for deep autoencoder."""
    print("""
    INPUT: [B, 3, 256, 256] - RGB Histology Image
         │
         ▼
    ┌─────────────────────────────────────────────────────────────────────────────────┐
    │                              ENCODER PATHWAY                                    │
    │                           (With Residual Blocks)                               │
    └─────────────────────────────────────────────────────────────────────────────────┘
         │
         ▼
    ENC1: [B, 3, 256, 256] → [B, 64, 128, 128]     ┐
         │ ResBlock + Downsample                   │ SKIP
         │ (Conv + BN + ReLU + Conv + BN) + Pool   │ CONNECTION
         ▼                                         │
    ENC2: [B, 64, 128, 128] → [B, 128, 64, 64]     │     ┐
         │ ResBlock + Downsample                   │     │ SKIP
         ▼                                         │     │ CONNECTION
    ENC3: [B, 128, 64, 64] → [B, 256, 32, 32]      │     │     ┐
         │ ResBlock + Downsample                   │     │     │ SKIP
         ▼                                         │     │     │ CONNECTION
    ENC4: [B, 256, 32, 32] → [B, 512, 16, 16]      │     │     │     ┐
         │ ResBlock + Downsample                   │     │     │     │ SKIP
         ▼                                         │     │     │     │ CONNECTION
    ENC5: [B, 512, 16, 16] → [B, 1024, 8, 8]       │     │     │     │     ┐
         │ ResBlock + Downsample                   │     │     │     │     │ SKIP
         ▼                                         │     │     │     │     │ CONNECTION
    ┌─────────────────────────────────────────────────────────────────────────────────┐
    │                              BOTTLENECK                                        │
    │ [B, 1024, 8, 8] → Multiple ResBlocks → [B, 1024, 8, 8]                       │
    │ Deeper feature processing with residual connections                             │
    └─────────────────────────────────────────────────────────────────────────────────┘
         │
         ▼
    ┌─────────────────────────────────────────────────────────────────────────────────┐
    │                              DECODER PATHWAY                                   │
    │                           (With Residual Blocks)                               │
    └─────────────────────────────────────────────────────────────────────────────────┘
         │
         ▼
    DEC5: [B, 1024, 8, 8] → [B, 512, 16, 16] ←──────┘     │     │     │     │
         │ ResBlock + Upsample + Skip                      │     │     │     │
         ▼                                                 │     │     │     │
    DEC4: [B, 1024, 16, 16] → [B, 256, 32, 32] ←─────────┘     │     │     │
         │ ResBlock + Upsample + Skip                            │     │     │
         ▼                                                       │     │     │
    DEC3: [B, 512, 32, 32] → [B, 128, 64, 64] ←───────────────┘     │     │
         │ ResBlock + Upsample + Skip                                │     │
         ▼                                                           │     │
    DEC2: [B, 256, 64, 64] → [B, 64, 128, 128] ←─────────────────────┘     │
         │ ResBlock + Upsample + Skip                                      │
         ▼                                                                 │
    DEC1: [B, 128, 128, 128] → [B, 64, 256, 256] ←─────────────────────────┘
         │ ResBlock + Upsample + Skip
         ▼
    ┌─────────────────────────────────────────────────────────────────────────────────┐
    │                              FINAL LAYER                                       │
    │ [B, 64, 256, 256] → [B, 3, 256, 256]                                          │
    │ Conv1x1 + Sigmoid                                                               │
    └─────────────────────────────────────────────────────────────────────────────────┘
         │
         ▼
    OUTPUT: [B, 3, 256, 256] - Denoised RGB Image
    
    Key:
    ResBlock = Residual Block (Conv + BN + ReLU + Conv + BN + Skip)
    Deeper network with more feature extraction capability
    Better for complex histopathology patterns
    """)


def print_parameter_breakdown(model_type="basic"):
    """
    Print detailed parameter breakdown by component.
    """
    print("=" * 100)
    print(f"PARAMETER BREAKDOWN - {model_type.upper()} AUTOENCODER")
    print("=" * 100)
    
    # Create model
    if model_type == "basic":
        model = HistoAutoencoder(input_channels=3, base_channels=64)
    elif model_type == "deep":
        model = DeepHistoAutoencoder(input_channels=3, base_channels=64)
    else:
        raise ValueError("model_type must be 'basic' or 'deep'")
    
    # Group parameters by component
    encoder_params = 0
    decoder_params = 0
    bottleneck_params = 0
    final_params = 0
    
    print("COMPONENT-WISE PARAMETER COUNT:")
    print("-" * 80)
    
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Leaf modules only
            param_count = sum(p.numel() for p in module.parameters())
            if param_count > 0:
                # Categorize based on name
                if 'encoder' in name:
                    encoder_params += param_count
                    component = "ENCODER"
                elif 'decoder' in name:
                    decoder_params += param_count
                    component = "DECODER"
                elif 'bottleneck' in name:
                    bottleneck_params += param_count
                    component = "BOTTLENECK"
                elif 'final' in name:
                    final_params += param_count
                    component = "FINAL"
                else:
                    component = "OTHER"
                
                print(f"{component:<12} {name:<40} {param_count:>12,}")
    
    print("-" * 80)
    print(f"{'ENCODER TOTAL:':<53} {encoder_params:>12,}")
    print(f"{'BOTTLENECK TOTAL:':<53} {bottleneck_params:>12,}")
    print(f"{'DECODER TOTAL:':<53} {decoder_params:>12,}")
    print(f"{'FINAL LAYER TOTAL:':<53} {final_params:>12,}")
    print("-" * 80)
    
    total = encoder_params + decoder_params + bottleneck_params + final_params
    print(f"{'GRAND TOTAL:':<53} {total:>12,}")
    
    # Print percentages
    print()
    print("PARAMETER DISTRIBUTION:")
    print("-" * 40)
    print(f"Encoder:    {encoder_params/total*100:5.1f}%")
    print(f"Bottleneck: {bottleneck_params/total*100:5.1f}%")
    print(f"Decoder:    {decoder_params/total*100:5.1f}%")
    print(f"Final:      {final_params/total*100:5.1f}%")
    print()


def print_memory_analysis(model_type="basic", batch_size=8):
    """
    Print memory usage analysis for training and inference.
    """
    print("=" * 100)
    print(f"MEMORY ANALYSIS - {model_type.upper()} AUTOENCODER")
    print("=" * 100)
    
    # Create model
    if model_type == "basic":
        model = HistoAutoencoder(input_channels=3, base_channels=64)
    elif model_type == "deep":
        model = DeepHistoAutoencoder(input_channels=3, base_channels=64)
    else:
        raise ValueError("model_type must be 'basic' or 'deep'")
    
    total_params, _ = count_parameters(model)
    
    print(f"Analysis for batch size: {batch_size}")
    print()
    
    # Model parameters memory
    model_memory = total_params * 4 / (1024 * 1024)  # 4 bytes per float32
    print(f"Model Parameters Memory: {model_memory:.2f} MB")
    
    # Input/Output memory
    input_memory = batch_size * 3 * 256 * 256 * 4 / (1024 * 1024)
    print(f"Input Tensor Memory: {input_memory:.2f} MB")
    print(f"Output Tensor Memory: {input_memory:.2f} MB")
    
    # Estimated intermediate activations (rough calculation)
    # This is a simplified estimation
    intermediate_memory = 0
    
    # Encoder activations
    for i, scale in enumerate([128, 64, 32, 16]):
        channels = 64 * (2 ** i)
        activation_memory = batch_size * channels * scale * scale * 4 / (1024 * 1024)
        intermediate_memory += activation_memory
        print(f"Encoder {i+1} Activations: {activation_memory:.2f} MB")
    
    # Bottleneck
    bottleneck_memory = batch_size * 1024 * 16 * 16 * 4 / (1024 * 1024)
    intermediate_memory += bottleneck_memory
    print(f"Bottleneck Activations: {bottleneck_memory:.2f} MB")
    
    # Decoder activations (similar to encoder)
    for i, scale in enumerate([32, 64, 128, 256]):
        channels = 512 // (2 ** i)
        activation_memory = batch_size * channels * scale * scale * 4 / (1024 * 1024)
        intermediate_memory += activation_memory
        print(f"Decoder {i+1} Activations: {activation_memory:.2f} MB")
    
    print()
    print(f"Total Intermediate Activations: {intermediate_memory:.2f} MB")
    
    # Training memory (gradients + optimizer states)
    gradient_memory = model_memory  # Same as parameters
    optimizer_memory = model_memory * 2  # Adam keeps running averages
    
    print()
    print("TRAINING MEMORY BREAKDOWN:")
    print("-" * 40)
    print(f"Model Parameters: {model_memory:.2f} MB")
    print(f"Gradients: {gradient_memory:.2f} MB")
    print(f"Optimizer States: {optimizer_memory:.2f} MB")
    print(f"Input/Output: {input_memory * 2:.2f} MB")
    print(f"Intermediate Activations: {intermediate_memory:.2f} MB")
    print("-" * 40)
    
    total_training_memory = model_memory + gradient_memory + optimizer_memory + input_memory * 2 + intermediate_memory
    print(f"Total Training Memory: {total_training_memory:.2f} MB")
    print(f"Total Training Memory: {total_training_memory/1024:.2f} GB")
    
    print()
    print("INFERENCE MEMORY BREAKDOWN:")
    print("-" * 40)
    inference_memory = model_memory + input_memory * 2 + intermediate_memory
    print(f"Total Inference Memory: {inference_memory:.2f} MB")
    print(f"Total Inference Memory: {inference_memory/1024:.2f} GB")
    
    print()
    print("RECOMMENDED GPU MEMORY:")
    print("-" * 40)
    recommended_memory = total_training_memory * 1.5  # 50% buffer
    print(f"Minimum for Training: {total_training_memory/1024:.1f} GB")
    print(f"Recommended for Training: {recommended_memory/1024:.1f} GB")
    print(f"Minimum for Inference: {inference_memory/1024:.1f} GB")


def compare_models():
    """
    Compare both autoencoder models side by side.
    """
    print("=" * 120)
    print("MODEL COMPARISON - BASIC vs DEEP AUTOENCODER")
    print("=" * 120)
    
    # Create both models
    basic_model = HistoAutoencoder(input_channels=3, base_channels=64)
    deep_model = DeepHistoAutoencoder(input_channels=3, base_channels=64)
    
    basic_params, _ = count_parameters(basic_model)
    deep_params, _ = count_parameters(deep_model)
    
    print(f"{'Metric':<30} {'Basic Autoencoder':<25} {'Deep Autoencoder':<25}")
    print("-" * 80)
    print(f"{'Architecture':<30} {'U-Net with skip conn.':<25} {'ResNet + U-Net':<25}")
    print(f"{'Total Parameters':<30} {f'{basic_params:,}':<25} {f'{deep_params:,}':<25}")
    print(f"{'Model Size (MB)':<30} {f'{basic_params*4/(1024*1024):.1f}':<25} {f'{deep_params*4/(1024*1024):.1f}':<25}")
    print(f"{'Depth':<30} {'8 layers':<25} {'12+ layers':<25}")
    print(f"{'Skip Connections':<30} {'Yes':<25} {'Yes + Residual':<25}")
    print(f"{'Best for':<30} {'Quick training':<25} {'Complex patterns':<25}")
    print(f"{'Training Time':<30} {'Faster':<25} {'Slower':<25}")
    print(f"{'Memory Usage':<30} {'Lower':<25} {'Higher':<25}")
    print(f"{'Recommended Use':<30} {'256x256 patches':<25} {'512x512+ patches':<25}")
    
    print()
    print("PERFORMANCE CHARACTERISTICS:")
    print("-" * 50)
    print("Basic Autoencoder:")
    print("  ✓ Faster training and inference")
    print("  ✓ Lower memory requirements")
    print("  ✓ Good for standard denoising tasks")
    print("  ✓ Suitable for limited GPU resources")
    print()
    print("Deep Autoencoder:")
    print("  ✓ Better feature extraction")
    print("  ✓ Handles complex tissue patterns")
    print("  ✓ Superior for high-resolution images")
    print("  ✓ More robust to various noise types")


def create_visualization_report(output_file="model_visualization_report.txt"):
    """
    Create a comprehensive report with all visualizations.
    """
    import sys
    from io import StringIO
    
    # Capture output
    old_stdout = sys.stdout
    sys.stdout = captured_output = StringIO()
    
    try:
        # Generate all visualizations
        print("HISTOPATHOLOGY AUTOENCODER - COMPREHENSIVE VISUALIZATION REPORT")
        print("Generated on:", __import__('datetime').datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        print("=" * 100)
        print()
        
        compare_models()
        print("\n\n")
        
        for model_type in ["basic", "deep"]:
            print_architecture_map(model_type)
            print("\n")
            print_data_flow_map(model_type)
            print("\n")
            print_parameter_breakdown(model_type)
            print("\n")
            print_memory_analysis(model_type)
            print("\n" + "="*100 + "\n")
        
    finally:
        sys.stdout = old_stdout
    
    # Write to file
    with open(output_file, 'w') as f:
        f.write(captured_output.getvalue())
    
    print(f"Comprehensive visualization report saved to: {output_file}")


def main():
    """
    Main function to demonstrate all visualization capabilities.
    """
    print("HISTOPATHOLOGY AUTOENCODER VISUALIZATION SYSTEM")
    print("=" * 60)
    print()
    
    while True:
        print("Available visualizations:")
        print("1. Model Comparison")
        print("2. Basic Autoencoder - Architecture Map")
        print("3. Basic Autoencoder - Data Flow Map")
        print("4. Basic Autoencoder - Parameter Breakdown")
        print("5. Basic Autoencoder - Memory Analysis")
        print("6. Deep Autoencoder - Architecture Map")
        print("7. Deep Autoencoder - Data Flow Map")
        print("8. Deep Autoencoder - Parameter Breakdown")
        print("9. Deep Autoencoder - Memory Analysis")
        print("10. Generate Complete Report")
        print("0. Exit")
        print()
        
        choice = input("Select option (0-10): ").strip()
        
        if choice == "0":
            break
        elif choice == "1":
            compare_models()
        elif choice == "2":
            print_architecture_map("basic")
        elif choice == "3":
            print_data_flow_map("basic")
        elif choice == "4":
            print_parameter_breakdown("basic")
        elif choice == "5":
            print_memory_analysis("basic")
        elif choice == "6":
            print_architecture_map("deep")
        elif choice == "7":
            print_data_flow_map("deep")
        elif choice == "8":
            print_parameter_breakdown("deep")
        elif choice == "9":
            print_memory_analysis("deep")
        elif choice == "10":
            create_visualization_report()
        else:
            print("Invalid choice. Please try again.")
        
        input("\nPress Enter to continue...")
        print("\n" + "="*60 + "\n")


if __name__ == "__main__":
    main()

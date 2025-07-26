import torch
import torch.nn as nn
from models import HistoAutoencoder

def debug_autoencoder_dimensions():
    """Debug the dimensions through the autoencoder to identify the mismatch"""
    
    model = HistoAutoencoder(input_channels=3, base_channels=64)
    test_input = torch.randn(1, 3, 256, 256)
    
    print("Input shape:", test_input.shape)
    print("\n=== ENCODER FORWARD PASS ===")
    
    # Encoder forward pass with dimension tracking
    x = test_input
    print(f"Initial input: {x.shape}")
    
    enc1 = model.encoder1(x)
    print(f"After encoder1: {enc1.shape}")
    
    enc2 = model.encoder2(enc1)
    print(f"After encoder2: {enc2.shape}")
    
    enc3 = model.encoder3(enc2)
    print(f"After encoder3: {enc3.shape}")
    
    enc4 = model.encoder4(enc3)
    print(f"After encoder4: {enc4.shape}")
    
    # Bottleneck
    bottleneck = model.bottleneck(enc4)
    print(f"After bottleneck: {bottleneck.shape}")
    
    print("\n=== DECODER FORWARD PASS ===")
    
    # Decoder forward pass
    dec4 = model.decoder4(bottleneck)
    print(f"After decoder4: {dec4.shape}")
    print(f"enc3 shape for skip connection: {enc3.shape}")
    
    # This is where the skip connection should work now
    try:
        dec4_skip = torch.cat([dec4, enc3], dim=1)  # Skip with enc3, not enc4
        print(f"After skip connection (dec4 + enc3): {dec4_skip.shape}")
    except RuntimeError as e:
        print(f"ERROR at skip connection: {e}")
        return
    
    dec3 = model.decoder3(dec4_skip)
    print(f"After decoder3: {dec3.shape}")
    print(f"enc2 shape for skip connection: {enc2.shape}")
    
    try:
        dec3_skip = torch.cat([dec3, enc2], dim=1)
        print(f"After skip connection (dec3 + enc2): {dec3_skip.shape}")
    except RuntimeError as e:
        print(f"ERROR at dec3 skip connection: {e}")
        return

if __name__ == "__main__":
    debug_autoencoder_dimensions()

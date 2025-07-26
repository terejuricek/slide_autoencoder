import torch
import torch.nn as nn
import torch.nn.functional as F


class HistoAutoencoder(nn.Module):
    """
    Autoencoder designed specifically for histopathological image denoising.
    Features:
    - Skip connections for better detail preservation
    - Batch normalization for stable training
    - LeakyReLU activations to prevent dead neurons
    - Appropriate for 256x256 or 512x512 histology patches
    """
    
    def __init__(self, input_channels=3, base_channels=64):
        super(HistoAutoencoder, self).__init__()
        
        # Encoder
        self.encoder1 = self._make_encoder_block(input_channels, base_channels)
        self.encoder2 = self._make_encoder_block(base_channels, base_channels * 2)
        self.encoder3 = self._make_encoder_block(base_channels * 2, base_channels * 4)
        self.encoder4 = self._make_encoder_block(base_channels * 4, base_channels * 8)
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(base_channels * 8, base_channels * 16, 3, padding=1),
            nn.BatchNorm2d(base_channels * 16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels * 16, base_channels * 16, 3, padding=1),
            nn.BatchNorm2d(base_channels * 16),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Decoder
        self.decoder4 = self._make_decoder_block(base_channels * 16, base_channels * 8)
        self.decoder3 = self._make_decoder_block(base_channels * 12, base_channels * 4)  # 8+4 due to skip connection
        self.decoder2 = self._make_decoder_block(base_channels * 6, base_channels * 2)   # 4+2 due to skip connection
        self.decoder1 = self._make_decoder_block(base_channels * 3, base_channels)       # 2+1 due to skip connection
        
        # Final output layer
        self.final = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, 3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels, input_channels, 1),
            nn.Sigmoid()  # Output values between 0 and 1
        )
        
    def _make_encoder_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(2, 2)
        )
    
    def _make_decoder_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 4, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )
    
    def forward(self, x):
        # Encoder with skip connections
        enc1 = self.encoder1(x)      # (64, 128, 128)
        enc2 = self.encoder2(enc1)   # (128, 64, 64)
        enc3 = self.encoder3(enc2)   # (256, 32, 32)
        enc4 = self.encoder4(enc3)   # (512, 16, 16)
        
        # Bottleneck
        bottleneck = self.bottleneck(enc4)  # (1024, 16, 16)
        
        # Decoder with skip connections
        dec4 = self.decoder4(bottleneck)    # (512, 32, 32)
        dec4 = torch.cat([dec4, enc3], dim=1)  # Skip connection with enc3, not enc4
        
        dec3 = self.decoder3(dec4)          # (256, 64, 64)
        dec3 = torch.cat([dec3, enc2], dim=1)  # Skip connection with enc2
        
        dec2 = self.decoder2(dec3)          # (128, 128, 128)
        dec2 = torch.cat([dec2, enc1], dim=1)  # Skip connection with enc1
        
        dec1 = self.decoder1(dec2)          # (64, 256, 256)
        # No skip connection for dec1 since we want original input size
        
        # Final output
        output = self.final(dec1)
        
        return output


class ResidualBlock(nn.Module):
    """Residual block for deeper autoencoder variants"""
    
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        
    def forward(self, x):
        residual = x
        out = F.leaky_relu(self.bn1(self.conv1(x)), 0.2)
        out = self.bn2(self.conv2(out))
        out += residual
        return F.leaky_relu(out, 0.2)


class DeepHistoAutoencoder(nn.Module):
    """
    Deeper autoencoder with residual connections for complex histology patterns
    """
    
    def __init__(self, input_channels=3, base_channels=64):
        super(DeepHistoAutoencoder, self).__init__()
        
        # Initial convolution
        self.initial = nn.Sequential(
            nn.Conv2d(input_channels, base_channels, 7, padding=3),
            nn.BatchNorm2d(base_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Encoder
        self.encoder1 = self._make_encoder_layer(base_channels, base_channels * 2, 2)
        self.encoder2 = self._make_encoder_layer(base_channels * 2, base_channels * 4, 2)
        self.encoder3 = self._make_encoder_layer(base_channels * 4, base_channels * 8, 2)
        self.encoder4 = self._make_encoder_layer(base_channels * 8, base_channels * 16, 2)
        
        # Bottleneck with residual blocks
        self.bottleneck = nn.Sequential(
            ResidualBlock(base_channels * 16),
            ResidualBlock(base_channels * 16),
            ResidualBlock(base_channels * 16)
        )
        
        # Decoder
        self.decoder4 = self._make_decoder_layer(base_channels * 16, base_channels * 8)
        self.decoder3 = self._make_decoder_layer(base_channels * 16, base_channels * 4)  # 8+8 from skip connection
        self.decoder2 = self._make_decoder_layer(base_channels * 8, base_channels * 2)   # 4+4 from skip connection  
        self.decoder1 = self._make_decoder_layer(base_channels * 4, base_channels)       # 2+2 from skip connection
        
        # Final output
        self.final = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, 3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels, input_channels, 1),
            nn.Sigmoid()
        )
        
    def _make_encoder_layer(self, in_channels, out_channels, num_blocks):
        layers = []
        layers.append(nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1))
        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        
        for _ in range(num_blocks):
            layers.append(ResidualBlock(out_channels))
            
        return nn.Sequential(*layers)
    
    def _make_decoder_layer(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 4, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
            ResidualBlock(out_channels)
        )
    
    def forward(self, x):
        # Initial processing
        x = self.initial(x)  # (64, 256, 256)
        
        # Encoder with skip connections
        enc1 = self.encoder1(x)   # (128, 128, 128)
        enc2 = self.encoder2(enc1) # (256, 64, 64)
        enc3 = self.encoder3(enc2) # (512, 32, 32)
        enc4 = self.encoder4(enc3) # (1024, 16, 16)
        
        # Bottleneck
        bottleneck = self.bottleneck(enc4)  # (1024, 16, 16)
        
        # Decoder with skip connections (corrected)
        dec4 = self.decoder4(bottleneck)    # (512, 32, 32)
        dec4 = torch.cat([dec4, enc3], dim=1)  # Skip with enc3
        
        dec3 = self.decoder3(dec4)          # (256, 64, 64)
        dec3 = torch.cat([dec3, enc2], dim=1)  # Skip with enc2
        
        dec2 = self.decoder2(dec3)          # (128, 128, 128)
        dec2 = torch.cat([dec2, enc1], dim=1)  # Skip with enc1
        
        dec1 = self.decoder1(dec2)          # (64, 256, 256)
        # Note: No skip with initial x since we want to maintain original input dimensions
        
        # Final output
        output = self.final(dec1)
        
        return output


def count_parameters(model):
    """Count the number of trainable parameters in the model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test the models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Test basic autoencoder
    model = HistoAutoencoder(input_channels=3, base_channels=64)
    model = model.to(device)
    print(f"Basic Autoencoder Parameters: {count_parameters(model):,}")
    
    # Test deep autoencoder
    deep_model = DeepHistoAutoencoder(input_channels=3, base_channels=64)
    deep_model = deep_model.to(device)
    print(f"Deep Autoencoder Parameters: {count_parameters(deep_model):,}")
    
    # Test forward pass
    test_input = torch.randn(1, 3, 256, 256).to(device)
    
    with torch.no_grad():
        output = model(test_input)
        print(f"Input shape: {test_input.shape}")
        print(f"Output shape: {output.shape}")
        
        deep_output = deep_model(test_input)
        print(f"Deep model output shape: {deep_output.shape}")

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional, Dict, List
import os
import time
from tqdm import tqdm
import json

from src.slide_autoencoder.models import HistoAutoencoder, DeepHistoAutoencoder
from src.slide_autoencoder.data_utils import get_data_loaders, denormalize_tensor


class HistologyTrainer:
    """
    Trainer class for histopathology autoencoder models
    """
    
    def __init__(self, 
                 model: nn.Module,
                 device: torch.device,
                 learning_rate: float = 1e-4,
                 weight_decay: float = 1e-5):
        """
        Initialize trainer
        
        Args:
            model: Autoencoder model to train
            device: Device to train on (cuda/cpu)
            learning_rate: Learning rate for optimizer
            weight_decay: Weight decay for regularization
        """
        self.model = model.to(device)
        self.device = device
        self.optimizer = optim.Adam(model.parameters(), 
                                   lr=learning_rate, 
                                   weight_decay=weight_decay)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10, verbose=True
        )
        
        # Loss functions
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        self.best_model_state = None
        
    def combined_loss(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Combined loss function using both MSE and L1 loss
        This helps preserve both overall structure and fine details
        """
        mse = self.mse_loss(output, target)
        l1 = self.l1_loss(output, target)
        return 0.7 * mse + 0.3 * l1
    
    def perceptual_loss(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Simple perceptual loss using gradient differences
        Helps preserve edges and textures important in histology
        """
        # Compute gradients
        def compute_gradients(img):
            dx = img[:, :, :, 1:] - img[:, :, :, :-1]
            dy = img[:, :, 1:, :] - img[:, :, :-1, :]
            return dx, dy
        
        dx_out, dy_out = compute_gradients(output)
        dx_tar, dy_tar = compute_gradients(target)
        
        grad_loss = self.mse_loss(dx_out, dx_tar) + self.mse_loss(dy_out, dy_tar)
        return grad_loss
    
    def train_epoch(self, train_loader: DataLoader) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = len(train_loader)
        
        progress_bar = tqdm(train_loader, desc="Training")
        
        for batch_idx, (noisy_images, clean_images) in enumerate(progress_bar):
            noisy_images = noisy_images.to(self.device)
            clean_images = clean_images.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            reconstructed = self.model(noisy_images)
            
            # Calculate loss
            reconstruction_loss = self.combined_loss(reconstructed, clean_images)
            perceptual_loss = self.perceptual_loss(reconstructed, clean_images)
            total_loss_batch = reconstruction_loss + 0.1 * perceptual_loss
            
            # Backward pass
            total_loss_batch.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            total_loss += total_loss_batch.item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f'{total_loss_batch.item():.4f}',
                'Avg Loss': f'{total_loss/(batch_idx+1):.4f}'
            })
        
        return total_loss / num_batches
    
    def validate_epoch(self, val_loader: DataLoader) -> float:
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0.0
        num_batches = len(val_loader)
        
        with torch.no_grad():
            for noisy_images, clean_images in tqdm(val_loader, desc="Validation"):
                noisy_images = noisy_images.to(self.device)
                clean_images = clean_images.to(self.device)
                
                reconstructed = self.model(noisy_images)
                
                reconstruction_loss = self.combined_loss(reconstructed, clean_images)
                perceptual_loss = self.perceptual_loss(reconstructed, clean_images)
                total_loss_batch = reconstruction_loss + 0.1 * perceptual_loss
                
                total_loss += total_loss_batch.item()
        
        return total_loss / num_batches
    
    def train(self, 
              train_loader: DataLoader,
              val_loader: Optional[DataLoader] = None,
              num_epochs: int = 100,
              save_dir: str = "checkpoints",
              save_every: int = 10) -> Dict[str, List[float]]:
        """
        Full training loop
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
            num_epochs: Number of epochs to train
            save_dir: Directory to save checkpoints
            save_every: Save model every N epochs
            
        Returns:
            Dictionary containing training history
        """
        os.makedirs(save_dir, exist_ok=True)
        
        print(f"Starting training for {num_epochs} epochs")
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        start_time = time.time()
        
        for epoch in range(num_epochs):
            epoch_start = time.time()
            
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print("-" * 50)
            
            # Training
            train_loss = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)
            
            # Validation
            if val_loader is not None:
                val_loss = self.validate_epoch(val_loader)
                self.val_losses.append(val_loss)
                
                # Learning rate scheduling
                self.scheduler.step(val_loss)
                
                print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
                
                # Save best model
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.best_model_state = self.model.state_dict().copy()
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.best_model_state,
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'train_loss': train_loss,
                        'val_loss': val_loss,
                    }, os.path.join(save_dir, 'best_model.pth'))
                    print(f"New best model saved! Val Loss: {val_loss:.4f}")
            else:
                print(f"Train Loss: {train_loss:.4f}")
            
            # Save checkpoint
            if (epoch + 1) % save_every == 0:
                checkpoint_path = os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss if val_loader else None,
                }, checkpoint_path)
                print(f"Checkpoint saved: {checkpoint_path}")
            
            epoch_time = time.time() - epoch_start
            print(f"Epoch time: {epoch_time:.2f}s")
        
        total_time = time.time() - start_time
        print(f"\nTraining completed in {total_time:.2f}s")
        
        # Save training history
        history = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }
        
        with open(os.path.join(save_dir, 'training_history.json'), 'w') as f:
            json.dump(history, f)
        
        return history
    
    def plot_training_history(self, save_path: Optional[str] = None):
        """Plot training and validation losses"""
        plt.figure(figsize=(10, 6))
        
        epochs = range(1, len(self.train_losses) + 1)
        plt.plot(epochs, self.train_losses, 'b-', label='Training Loss')
        
        if self.val_losses:
            plt.plot(epochs, self.val_losses, 'r-', label='Validation Loss')
        
        plt.title('Training History')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def evaluate_model(self, test_loader: DataLoader, num_samples: int = 8):
        """
        Evaluate model and visualize results
        """
        self.model.eval()
        
        # Calculate metrics
        total_mse = 0.0
        total_psnr = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for noisy_images, clean_images in test_loader:
                noisy_images = noisy_images.to(self.device)
                clean_images = clean_images.to(self.device)
                
                reconstructed = self.model(noisy_images)
                
                # Calculate metrics
                mse = self.mse_loss(reconstructed, clean_images)
                psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
                
                total_mse += mse.item()
                total_psnr += psnr.item()
                num_batches += 1
        
        avg_mse = total_mse / num_batches
        avg_psnr = total_psnr / num_batches
        
        print(f"Average MSE: {avg_mse:.6f}")
        print(f"Average PSNR: {avg_psnr:.2f} dB")
        
        # Visualize results
        self._visualize_results(test_loader, num_samples)
        
        return avg_mse, avg_psnr
    
    def _visualize_results(self, test_loader: DataLoader, num_samples: int = 8):
        """Visualize reconstruction results"""
        self.model.eval()
        
        dataiter = iter(test_loader)
        noisy_images, clean_images = next(dataiter)
        
        with torch.no_grad():
            noisy_images = noisy_images.to(self.device)
            reconstructed = self.model(noisy_images)
            
            # Move to CPU and denormalize
            noisy_images = noisy_images.cpu()
            clean_images = clean_images.cpu()
            reconstructed = reconstructed.cpu()
        
        # Plot results
        fig, axes = plt.subplots(3, num_samples, figsize=(20, 8))
        
        for i in range(min(num_samples, noisy_images.size(0))):
            # Denormalize images
            noisy_img = denormalize_tensor(noisy_images[i])
            clean_img = denormalize_tensor(clean_images[i])
            recon_img = denormalize_tensor(reconstructed[i])
            
            # Convert to numpy
            noisy_np = noisy_img.permute(1, 2, 0).numpy()
            clean_np = clean_img.permute(1, 2, 0).numpy()
            recon_np = recon_img.permute(1, 2, 0).numpy()
            
            axes[0, i].imshow(noisy_np)
            axes[0, i].set_title('Noisy Input')
            axes[0, i].axis('off')
            
            axes[1, i].imshow(recon_np)
            axes[1, i].set_title('Reconstructed')
            axes[1, i].axis('off')
            
            axes[2, i].imshow(clean_np)
            axes[2, i].set_title('Ground Truth')
            axes[2, i].axis('off')
        
        plt.tight_layout()
        plt.show()


def train_autoencoder(data_dir: str,
                     model_type: str = "basic",
                     batch_size: int = 16,
                     num_epochs: int = 100,
                     learning_rate: float = 1e-4,
                     patch_size: int = 256):
    """
    Main training function
    
    Args:
        data_dir: Directory containing training images
        model_type: Type of model ("basic" or "deep")
        batch_size: Batch size for training
        num_epochs: Number of training epochs
        learning_rate: Learning rate
        patch_size: Size of image patches
    """
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create model
    if model_type == "basic":
        model = HistoAutoencoder(input_channels=3, base_channels=64)
    elif model_type == "deep":
        model = DeepHistoAutoencoder(input_channels=3, base_channels=64)
    else:
        raise ValueError("model_type must be 'basic' or 'deep'")
    
    print(f"Model: {model.__class__.__name__}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create data loaders
    train_loader, val_loader = get_data_loaders(
        train_dir=data_dir,
        batch_size=batch_size,
        patch_size=patch_size,
        noise_factor=0.3
    )
    
    # Create trainer
    trainer = HistologyTrainer(model, device, learning_rate)
    
    # Train model
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=num_epochs,
        save_dir="checkpoints"
    )
    
    # Plot training history
    trainer.plot_training_history("training_history.png")
    
    # Evaluate model
    if val_loader:
        trainer.evaluate_model(val_loader)
    
    return trainer, history


if __name__ == "__main__":
    # Example usage
    data_dir = "data/synthetic"  # Replace with your data directory
    
    trainer, history = train_autoencoder(
        data_dir=data_dir,
        model_type="basic",
        batch_size=8,
        num_epochs=50,
        learning_rate=1e-4,
        patch_size=256
    )

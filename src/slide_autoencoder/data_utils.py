import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import numpy as np
import cv2
from PIL import Image
import os
import random
from typing import Tuple, Optional, List
import matplotlib.pyplot as plt


class HistologyDataset(Dataset):
    """
    Dataset class for histopathological images with noise augmentation
    """
    
    def __init__(self, 
                 image_dir: str, 
                 patch_size: int = 256,
                 noise_factor: float = 0.3,
                 transform: Optional[transforms.Compose] = None):
        """
        Args:
            image_dir: Directory containing histology images
            patch_size: Size of image patches (assumes square patches)
            noise_factor: Factor for adding noise (0.0 to 1.0)
            transform: Optional transforms to apply
        """
        self.image_dir = image_dir
        self.patch_size = patch_size
        self.noise_factor = noise_factor
        self.image_files = [f for f in os.listdir(image_dir) 
                           if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.tif'))]
        
        # Default transforms for histology images
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((patch_size, patch_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])  # ImageNet normalization
            ])
        else:
            self.transform = transform
            
        # Noise transform (applied after normalization)
        self.noise_transform = transforms.Compose([
            transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225], 
                               std=[1/0.229, 1/0.224, 1/0.225]),  # Denormalize
            transforms.Lambda(lambda x: torch.clamp(x, 0, 1))  # Ensure valid range
        ])
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # Load image
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        image = Image.open(img_path).convert('RGB')
        
        # Apply transforms
        clean_image = self.transform(image)
        
        # Create noisy version
        noisy_image = self._add_noise(clean_image.clone())
        
        return noisy_image, clean_image
    
    def _add_noise(self, image: torch.Tensor) -> torch.Tensor:
        """Add various types of noise to simulate real-world conditions"""
        # Denormalize for noise addition
        image = self.noise_transform(image)
        
        # Gaussian noise
        if random.random() < 0.5:
            noise = torch.randn_like(image) * self.noise_factor * 0.1
            image = image + noise
        
        # Salt and pepper noise
        if random.random() < 0.3:
            mask = torch.rand_like(image) < 0.02
            image[mask] = torch.rand_like(image[mask])
        
        # Blur (motion blur or defocus)
        if random.random() < 0.3:
            # Convert to numpy for OpenCV processing
            img_np = image.permute(1, 2, 0).numpy()
            if random.random() < 0.5:
                # Motion blur
                kernel_size = random.choice([3, 5, 7])
                kernel = np.zeros((kernel_size, kernel_size))
                kernel[int((kernel_size-1)/2), :] = np.ones(kernel_size)
                kernel = kernel / kernel_size
                img_np = cv2.filter2D(img_np, -1, kernel)
            else:
                # Gaussian blur
                kernel_size = random.choice([3, 5])
                img_np = cv2.GaussianBlur(img_np, (kernel_size, kernel_size), 0)
            
            image = torch.from_numpy(img_np).permute(2, 0, 1)
        
        # Clamp values and renormalize
        image = torch.clamp(image, 0, 1)
        image = transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])(image)
        
        return image


def create_synthetic_patches(output_dir: str, 
                           num_patches: int = 1000, 
                           patch_size: int = 256):
    """
    Create synthetic histology-like patches for testing
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for i in range(num_patches):
        # Create synthetic tissue-like patterns
        img = np.zeros((patch_size, patch_size, 3), dtype=np.uint8)
        
        # Background (pink/purple staining)
        bg_color = np.random.randint(200, 255, 3)
        img[:, :] = bg_color
        
        # Add cell nuclei (dark blue/purple spots)
        num_nuclei = np.random.randint(5, 20)
        for _ in range(num_nuclei):
            center = (np.random.randint(20, patch_size-20), 
                     np.random.randint(20, patch_size-20))
            radius = np.random.randint(3, 8)
            color = np.random.randint(0, 100, 3)
            cv2.circle(img, center, radius, color.tolist(), -1)
        
        # Add some texture
        noise = np.random.normal(0, 10, img.shape).astype(np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        # Save image
        filename = f"synthetic_patch_{i:04d}.png"
        cv2.imwrite(os.path.join(output_dir, filename), 
                   cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    
    print(f"Created {num_patches} synthetic patches in {output_dir}")


def get_data_loaders(train_dir: str, 
                    val_dir: Optional[str] = None,
                    batch_size: int = 16,
                    patch_size: int = 256,
                    noise_factor: float = 0.3,
                    num_workers: int = 4) -> Tuple[DataLoader, Optional[DataLoader]]:
    """
    Create data loaders for training and validation
    """
    
    # Training transforms with augmentation
    train_transform = transforms.Compose([
        transforms.Resize((patch_size, patch_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=90),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Validation transforms (no augmentation)
    val_transform = transforms.Compose([
        transforms.Resize((patch_size, patch_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    train_dataset = HistologyDataset(train_dir, patch_size, noise_factor, train_transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                             shuffle=True, num_workers=num_workers, pin_memory=True)
    
    val_loader = None
    if val_dir and os.path.exists(val_dir):
        val_dataset = HistologyDataset(val_dir, patch_size, noise_factor, val_transform)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, 
                               shuffle=False, num_workers=num_workers, pin_memory=True)
    
    return train_loader, val_loader


def denormalize_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """
    Denormalize tensor for visualization
    """
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    
    if tensor.is_cuda:
        mean = mean.cuda()
        std = std.cuda()
    
    tensor = tensor * std + mean
    return torch.clamp(tensor, 0, 1)


def visualize_samples(dataloader: DataLoader, num_samples: int = 4):
    """
    Visualize some samples from the dataloader
    """
    dataiter = iter(dataloader)
    noisy_images, clean_images = next(dataiter)
    
    fig, axes = plt.subplots(2, num_samples, figsize=(15, 6))
    
    for i in range(min(num_samples, noisy_images.size(0))):
        # Denormalize images
        noisy_img = denormalize_tensor(noisy_images[i])
        clean_img = denormalize_tensor(clean_images[i])
        
        # Convert to numpy and transpose
        noisy_np = noisy_img.permute(1, 2, 0).numpy()
        clean_np = clean_img.permute(1, 2, 0).numpy()
        
        axes[0, i].imshow(noisy_np)
        axes[0, i].set_title('Noisy')
        axes[0, i].axis('off')
        
        axes[1, i].imshow(clean_np)
        axes[1, i].set_title('Clean')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Create synthetic data for testing
    synthetic_dir = "data/synthetic"
    create_synthetic_patches(synthetic_dir, num_patches=100, patch_size=256)
    
    # Test data loading
    train_loader, _ = get_data_loaders(synthetic_dir, batch_size=4)
    
    print(f"Created dataloader with {len(train_loader)} batches")
    print(f"Each batch contains {train_loader.batch_size} samples")
    
    # Visualize some samples
    try:
        visualize_samples(train_loader, num_samples=4)
    except Exception as e:
        print(f"Visualization requires matplotlib GUI backend: {e}")
        print("Run this in a Jupyter notebook or with proper display for visualization")

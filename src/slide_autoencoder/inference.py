import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import os
from typing import Tuple, Optional

from .models import HistoAutoencoder, DeepHistoAutoencoder
from .data_utils import denormalize_tensor
import torchvision.transforms as transforms


class HistologyDenoiser:
    """
    Inference class for denoising histopathological images
    """
    
    def __init__(self, model_path: str, model_type: str = "basic", device: Optional[torch.device] = None):
        """
        Initialize the denoiser
        
        Args:
            model_path: Path to the trained model checkpoint
            model_type: Type of model ("basic" or "deep")
            device: Device to run inference on
        """
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load model
        if model_type == "basic":
            self.model = HistoAutoencoder(input_channels=3, base_channels=64)
        elif model_type == "deep":
            self.model = DeepHistoAutoencoder(input_channels=3, base_channels=64)
        else:
            raise ValueError("model_type must be 'basic' or 'deep'")
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        
        self.model.to(self.device)
        self.model.eval()
        
        # Preprocessing transforms
        self.preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        print(f"Loaded model: {model_type} autoencoder")
        print(f"Device: {self.device}")
    
    def denoise_image(self, image: np.ndarray, patch_size: int = 256, overlap: int = 32) -> np.ndarray:
        """
        Denoise a full histopathological image using sliding window approach
        
        Args:
            image: Input image as numpy array (H, W, C)
            patch_size: Size of patches for processing
            overlap: Overlap between patches
            
        Returns:
            Denoised image as numpy array
        """
        if len(image.shape) != 3 or image.shape[2] != 3:
            raise ValueError("Input image must be RGB (H, W, 3)")
        
        height, width, channels = image.shape
        stride = patch_size - overlap
        
        # Pad image to handle edge cases
        pad_h = stride - (height - patch_size) % stride if (height - patch_size) % stride != 0 else 0
        pad_w = stride - (width - patch_size) % stride if (width - patch_size) % stride != 0 else 0
        
        padded_image = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
        padded_height, padded_width = padded_image.shape[:2]
        
        # Initialize output and weight maps
        output = np.zeros_like(padded_image, dtype=np.float32)
        weight_map = np.zeros(padded_image.shape[:2], dtype=np.float32)
        
        # Process patches
        with torch.no_grad():
            for y in range(0, padded_height - patch_size + 1, stride):
                for x in range(0, padded_width - patch_size + 1, stride):
                    # Extract patch
                    patch = padded_image[y:y+patch_size, x:x+patch_size]
                    
                    # Convert to PIL and preprocess
                    patch_pil = Image.fromarray(patch.astype(np.uint8))
                    patch_tensor = self.preprocess(patch_pil).unsqueeze(0).to(self.device)
                    
                    # Denoise patch
                    denoised_tensor = self.model(patch_tensor)
                    
                    # Denormalize and convert back to numpy
                    denoised_tensor = denormalize_tensor(denoised_tensor.cpu().squeeze(0))
                    denoised_patch = denoised_tensor.permute(1, 2, 0).numpy()
                    denoised_patch = (denoised_patch * 255).astype(np.uint8)
                    
                    # Create weight mask (higher weights in center)
                    weight_patch = self._create_weight_mask(patch_size, overlap)
                    
                    # Add to output with weighting
                    output[y:y+patch_size, x:x+patch_size] += denoised_patch * weight_patch[..., None]
                    weight_map[y:y+patch_size, x:x+patch_size] += weight_patch
        
        # Normalize by weights
        weight_map[weight_map == 0] = 1  # Avoid division by zero
        output = output / weight_map[..., None]
        
        # Remove padding and return
        output = output[:height, :width].astype(np.uint8)
        return output
    
    def _create_weight_mask(self, patch_size: int, overlap: int) -> np.ndarray:
        """Create a weight mask with higher weights in the center"""
        mask = np.ones((patch_size, patch_size), dtype=np.float32)
        
        if overlap > 0:
            # Create fade-out regions at edges
            fade_region = overlap // 2
            
            # Top and bottom edges
            for i in range(fade_region):
                weight = (i + 1) / fade_region
                mask[i, :] *= weight
                mask[-(i+1), :] *= weight
            
            # Left and right edges
            for i in range(fade_region):
                weight = (i + 1) / fade_region
                mask[:, i] *= weight
                mask[:, -(i+1)] *= weight
        
        return mask
    
    def denoise_single_patch(self, image: np.ndarray, target_size: int = 256) -> np.ndarray:
        """
        Denoise a single image patch
        
        Args:
            image: Input image patch as numpy array (H, W, C)
            target_size: Target size for processing
            
        Returns:
            Denoised image patch
        """
        original_shape = image.shape[:2]
        
        # Resize if necessary
        if image.shape[0] != target_size or image.shape[1] != target_size:
            image = cv2.resize(image, (target_size, target_size))
        
        # Convert to PIL and preprocess
        image_pil = Image.fromarray(image.astype(np.uint8))
        image_tensor = self.preprocess(image_pil).unsqueeze(0).to(self.device)
        
        # Denoise
        with torch.no_grad():
            denoised_tensor = self.model(image_tensor)
        
        # Denormalize and convert back to numpy
        denoised_tensor = denormalize_tensor(denoised_tensor.cpu().squeeze(0))
        denoised_image = denoised_tensor.permute(1, 2, 0).numpy()
        denoised_image = (denoised_image * 255).astype(np.uint8)
        
        # Resize back to original size if necessary
        if denoised_image.shape[:2] != original_shape:
            denoised_image = cv2.resize(denoised_image, (original_shape[1], original_shape[0]))
        
        return denoised_image
    
    def batch_denoise_directory(self, input_dir: str, output_dir: str, patch_size: int = 256):
        """
        Denoise all images in a directory
        
        Args:
            input_dir: Directory containing input images
            output_dir: Directory to save denoised images
            patch_size: Size of patches for processing
        """
        os.makedirs(output_dir, exist_ok=True)
        
        image_extensions = ('.png', '.jpg', '.jpeg', '.tiff', '.tif', '.bmp')
        image_files = [f for f in os.listdir(input_dir) 
                      if f.lower().endswith(image_extensions)]
        
        print(f"Processing {len(image_files)} images...")
        
        for filename in image_files:
            print(f"Processing: {filename}")
            
            # Load image
            input_path = os.path.join(input_dir, filename)
            image = cv2.imread(input_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Denoise
            if max(image.shape[:2]) > patch_size * 2:
                # Use sliding window for large images
                denoised = self.denoise_image(image, patch_size=patch_size)
            else:
                # Process as single patch for small images
                denoised = self.denoise_single_patch(image, target_size=patch_size)
            
            # Save result
            output_path = os.path.join(output_dir, filename)
            denoised_bgr = cv2.cvtColor(denoised, cv2.COLOR_RGB2BGR)
            cv2.imwrite(output_path, denoised_bgr)
        
        print(f"Denoising complete. Results saved to: {output_dir}")
    
    def compare_results(self, original_image: np.ndarray, denoised_image: np.ndarray):
        """
        Visualize comparison between original and denoised images
        
        Args:
            original_image: Original noisy image
            denoised_image: Denoised image
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image
        axes[0].imshow(original_image)
        axes[0].set_title('Original (Noisy)')
        axes[0].axis('off')
        
        # Denoised image
        axes[1].imshow(denoised_image)
        axes[1].set_title('Denoised')
        axes[1].axis('off')
        
        # Difference map
        diff = np.abs(original_image.astype(np.float32) - denoised_image.astype(np.float32))
        diff = (diff / diff.max() * 255).astype(np.uint8)
        axes[2].imshow(diff, cmap='hot')
        axes[2].set_title('Difference Map')
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def calculate_metrics(self, original: np.ndarray, denoised: np.ndarray, 
                         ground_truth: Optional[np.ndarray] = None) -> dict:
        """
        Calculate image quality metrics
        
        Args:
            original: Original noisy image
            denoised: Denoised image
            ground_truth: Ground truth clean image (optional)
            
        Returns:
            Dictionary of metrics
        """
        metrics = {}
        
        # PSNR between original and denoised
        mse_orig = np.mean((original.astype(np.float32) - denoised.astype(np.float32)) ** 2)
        if mse_orig > 0:
            psnr_orig = 20 * np.log10(255.0 / np.sqrt(mse_orig))
            metrics['psnr_improvement'] = psnr_orig
        
        # If ground truth is available
        if ground_truth is not None:
            # PSNR with ground truth
            mse_gt = np.mean((ground_truth.astype(np.float32) - denoised.astype(np.float32)) ** 2)
            if mse_gt > 0:
                psnr_gt = 20 * np.log10(255.0 / np.sqrt(mse_gt))
                metrics['psnr_vs_gt'] = psnr_gt
            
            # SSIM (simplified version)
            metrics['mse_vs_gt'] = mse_gt
        
        return metrics


def demo_inference():
    """
    Demo function showing how to use the denoiser
    """
    # Initialize denoiser (replace with your model path)
    model_path = "checkpoints/best_model.pth"
    
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        print("Please train a model first or provide the correct path")
        return
    
    denoiser = HistologyDenoiser(model_path, model_type="basic")
    
    # Demo with synthetic data
    test_dir = "data/synthetic"
    output_dir = "results/denoised"
    
    if os.path.exists(test_dir):
        denoiser.batch_denoise_directory(test_dir, output_dir)
    else:
        print(f"Test directory {test_dir} not found")
        print("Please create test data first")


if __name__ == "__main__":
    demo_inference()

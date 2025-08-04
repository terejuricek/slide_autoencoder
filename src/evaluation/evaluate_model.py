#!/usr/bin/env python3
"""
Model Evaluation Script for Trained Autoencoder

This script evaluates a trained autoencoder model on validation or test data,
providing comprehensive metrics and visualizations.
"""

import torch
import torch.nn as nn
import numpy as np
import cv2
import os
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image
import json
from datetime import datetime
import argparse

from src.slide_autoencoder.models import HistoAutoencoder, DeepHistoAutoencoder
from src.slide_autoencoder.inference import HistologyDenoiser
from src.slide_autoencoder.data_utils import get_data_loaders


def calculate_metrics(original, denoised, ground_truth=None):
    """
    Calculate comprehensive quality metrics
    
    Args:
        original: Original noisy image
        denoised: Denoised image
        ground_truth: Clean ground truth (if available)
    
    Returns:
        Dictionary of metrics
    """
    metrics = {}
    
    # Convert to numpy if torch tensors
    if torch.is_tensor(original):
        original = original.detach().cpu().numpy()
    if torch.is_tensor(denoised):
        denoised = denoised.detach().cpu().numpy()
    if ground_truth is not None and torch.is_tensor(ground_truth):
        ground_truth = ground_truth.detach().cpu().numpy()
    
    # Ensure images are in range [0, 1]
    original = np.clip(original, 0, 1)
    denoised = np.clip(denoised, 0, 1)
    if ground_truth is not None:
        ground_truth = np.clip(ground_truth, 0, 1)
    
    # MSE and PSNR between original and denoised
    mse_orig = np.mean((original - denoised) ** 2)
    if mse_orig > 0:
        psnr_orig = 20 * np.log10(1.0 / np.sqrt(mse_orig))
    else:
        psnr_orig = float('inf')
    
    metrics['mse_vs_original'] = float(mse_orig)
    metrics['psnr_vs_original'] = float(psnr_orig)
    
    # Metrics vs ground truth if available
    if ground_truth is not None:
        # MSE and PSNR vs ground truth
        mse_gt = np.mean((ground_truth - denoised) ** 2)
        if mse_gt > 0:
            psnr_gt = 20 * np.log10(1.0 / np.sqrt(mse_gt))
        else:
            psnr_gt = float('inf')
        
        metrics['mse_vs_gt'] = float(mse_gt)
        metrics['psnr_vs_gt'] = float(psnr_gt)
        
        # SSIM (simplified implementation)
        ssim_gt = calculate_ssim(ground_truth, denoised)
        metrics['ssim_vs_gt'] = float(ssim_gt)
        
        # Improvement metrics
        mse_original_vs_gt = np.mean((ground_truth - original) ** 2)
        improvement = (mse_original_vs_gt - mse_gt) / mse_original_vs_gt * 100
        metrics['mse_improvement_percent'] = float(improvement)
    
    return metrics


def calculate_ssim(img1, img2, window_size=11, C1=0.01**2, C2=0.03**2):
    """
    Calculate Structural Similarity Index (SSIM) between two images
    Simplified implementation for grayscale or RGB images
    """
    # Convert to grayscale if RGB
    if len(img1.shape) == 3:
        img1 = np.mean(img1, axis=2)
        img2 = np.mean(img2, axis=2)
    
    # Calculate means
    mu1 = cv2.GaussianBlur(img1, (window_size, window_size), 1.5)
    mu2 = cv2.GaussianBlur(img2, (window_size, window_size), 1.5)
    
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2
    
    # Calculate variances and covariance
    sigma1_sq = cv2.GaussianBlur(img1 * img1, (window_size, window_size), 1.5) - mu1_sq
    sigma2_sq = cv2.GaussianBlur(img2 * img2, (window_size, window_size), 1.5) - mu2_sq
    sigma12 = cv2.GaussianBlur(img1 * img2, (window_size, window_size), 1.5) - mu1_mu2
    
    # Calculate SSIM
    numerator = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
    denominator = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    
    ssim_map = numerator / denominator
    return np.mean(ssim_map)


def evaluate_model(model_path, test_dir, output_dir, model_type="basic", device="auto"):
    """
    Comprehensive model evaluation
    
    Args:
        model_path: Path to trained model checkpoint
        test_dir: Directory with test images
        output_dir: Directory to save evaluation results
        model_type: "basic" or "deep"
        device: Device to use for evaluation
    """
    print(" Autoencoder Model Evaluation")
    print("=" * 50)
    
    # Setup device
    if device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
    
    print(f"📱 Using device: {device}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model
    print(f"📂 Loading model from {model_path}")
    denoiser = HistologyDenoiser(model_path, model_type=model_type, device=device)
    
    # Find test images
    image_extensions = ['.png', '.jpg', '.jpeg', '.tiff', '.tif']
    test_images = []
    for ext in image_extensions:
        test_images.extend(Path(test_dir).glob(f'*{ext}'))
        test_images.extend(Path(test_dir).glob(f'*{ext.upper()}'))
    
    print(f" Found {len(test_images)} test images")
    
    if len(test_images) == 0:
        print(" No test images found!")
        return
    
    # Evaluation storage
    all_metrics = []
    results = {
        'model_path': model_path,
        'model_type': model_type,
        'test_dir': test_dir,
        'num_images': len(test_images),
        'evaluation_date': datetime.now().isoformat(),
        'device': str(device)
    }
    
    # Process each image
    print(f" Processing {len(test_images)} images...")
    
    for i, img_path in enumerate(test_images):
        print(f"  Processing {img_path.name} ({i+1}/{len(test_images)})")
        
        try:
            # Load image
            image = cv2.imread(str(img_path))
            if image is None:
                print(f"     Could not load {img_path.name}")
                continue
            
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            original_image = image.copy()
            
            # Add noise for evaluation (simulate real noise)
            noise_factor = 0.1  # Light noise for evaluation
            noisy_image = image.astype(np.float32) / 255.0
            noise = np.random.normal(0, noise_factor, noisy_image.shape)
            noisy_image = np.clip(noisy_image + noise, 0, 1)
            noisy_image = (noisy_image * 255).astype(np.uint8)
            
            # Denoise
            denoised_image = denoiser.denoise_image(noisy_image, patch_size=256)
            
            # Calculate metrics (using original as ground truth)
            metrics = calculate_metrics(
                original=noisy_image.astype(np.float32) / 255.0,
                denoised=denoised_image.astype(np.float32) / 255.0,
                ground_truth=original_image.astype(np.float32) / 255.0
            )
            
            metrics['image_name'] = img_path.name
            metrics['image_size'] = f"{image.shape[1]}x{image.shape[0]}"
            all_metrics.append(metrics)
            
            # Save comparison images (for first 10 images)
            if i < 10:
                save_comparison(
                    original=original_image,
                    noisy=noisy_image,
                    denoised=denoised_image,
                    metrics=metrics,
                    save_path=Path(output_dir) / f"comparison_{i+1:02d}_{img_path.stem}.png"
                )
        
        except Exception as e:
            print(f"     Error processing {img_path.name}: {e}")
            continue
    
    # Calculate summary statistics
    if all_metrics:
        summary_stats = calculate_summary_stats(all_metrics)
        results['summary_metrics'] = summary_stats
        results['individual_metrics'] = all_metrics
        
        # Print summary
        print_evaluation_summary(summary_stats)
        
        # Save detailed results
        results_file = Path(output_dir) / "evaluation_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n Results saved to {results_file}")
        
        # Create visualizations
        create_evaluation_plots(all_metrics, output_dir)
        
        return results
    else:
        print(" No images could be processed!")
        return None


def save_comparison(original, noisy, denoised, metrics, save_path):
    """Save before/after comparison image"""
    
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    # Original
    axes[0].imshow(original)
    axes[0].set_title('Original (Ground Truth)')
    axes[0].axis('off')
    
    # Noisy
    axes[1].imshow(noisy)
    axes[1].set_title('Noisy Input')
    axes[1].axis('off')
    
    # Denoised
    axes[2].imshow(denoised)
    axes[2].set_title('Denoised Output')
    axes[2].axis('off')
    
    # Difference map
    diff = np.abs(original.astype(np.float32) - denoised.astype(np.float32))
    axes[3].imshow(diff, cmap='hot')
    axes[3].set_title('Absolute Difference')
    axes[3].axis('off')
    
    # Add metrics as text
    metrics_text = f"PSNR vs GT: {metrics.get('psnr_vs_gt', 0):.2f} dB\n"
    metrics_text += f"MSE Improvement: {metrics.get('mse_improvement_percent', 0):.1f}%\n"
    metrics_text += f"SSIM: {metrics.get('ssim_vs_gt', 0):.3f}"
    
    fig.suptitle(f"Denoising Results\n{metrics_text}", fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def calculate_summary_stats(metrics_list):
    """Calculate summary statistics from all metrics"""
    
    # Extract numeric metrics
    numeric_metrics = [
        'psnr_vs_gt', 'mse_vs_gt', 'ssim_vs_gt', 
        'mse_improvement_percent', 'psnr_vs_original'
    ]
    
    summary = {}
    
    for metric in numeric_metrics:
        values = [m.get(metric, 0) for m in metrics_list if metric in m]
        if values:
            summary[f"{metric}_mean"] = np.mean(values)
            summary[f"{metric}_std"] = np.std(values)
            summary[f"{metric}_min"] = np.min(values)
            summary[f"{metric}_max"] = np.max(values)
    
    return summary


def print_evaluation_summary(summary_stats):
    """Print evaluation summary to console"""
    
    print("\n Evaluation Summary")
    print("=" * 50)
    
    # PSNR vs Ground Truth
    if 'psnr_vs_gt_mean' in summary_stats:
        print(f" PSNR vs Ground Truth:")
        print(f"   Mean: {summary_stats['psnr_vs_gt_mean']:.2f} ± {summary_stats['psnr_vs_gt_std']:.2f} dB")
        print(f"   Range: {summary_stats['psnr_vs_gt_min']:.2f} - {summary_stats['psnr_vs_gt_max']:.2f} dB")
    
    # MSE Improvement
    if 'mse_improvement_percent_mean' in summary_stats:
        print(f"\nMSE Improvement:")
        print(f"   Mean: {summary_stats['mse_improvement_percent_mean']:.1f} ± {summary_stats['mse_improvement_percent_std']:.1f}%")
        print(f"   Range: {summary_stats['mse_improvement_percent_min']:.1f} - {summary_stats['mse_improvement_percent_max']:.1f}%")
    
    # SSIM
    if 'ssim_vs_gt_mean' in summary_stats:
        print(f"\n SSIM (Structural Similarity):")
        print(f"   Mean: {summary_stats['ssim_vs_gt_mean']:.3f} ± {summary_stats['ssim_vs_gt_std']:.3f}")
        print(f"   Range: {summary_stats['ssim_vs_gt_min']:.3f} - {summary_stats['ssim_vs_gt_max']:.3f}")
    
    # Quality assessment
    print(f"\nQuality Assessment:")
    psnr_mean = summary_stats.get('psnr_vs_gt_mean', 0)
    improvement_mean = summary_stats.get('mse_improvement_percent_mean', 0)
    
    if psnr_mean > 30:
        print("    Excellent denoising quality (PSNR > 30 dB)")
    elif psnr_mean > 25:
        print("    Good denoising quality (PSNR > 25 dB)")
    elif psnr_mean > 20:
        print("     Fair denoising quality (PSNR > 20 dB)")
    else:
        print("    Poor denoising quality (PSNR < 20 dB)")
    
    if improvement_mean > 50:
        print("    Significant noise reduction achieved")
    elif improvement_mean > 20:
        print("    Moderate noise reduction achieved")
    else:
        print("     Limited noise reduction achieved")


def create_evaluation_plots(metrics_list, output_dir):
    """Create evaluation plots and charts"""
    
    # Extract data for plotting
    psnr_values = [m.get('psnr_vs_gt', 0) for m in metrics_list if 'psnr_vs_gt' in m]
    improvement_values = [m.get('mse_improvement_percent', 0) for m in metrics_list if 'mse_improvement_percent' in m]
    ssim_values = [m.get('ssim_vs_gt', 0) for m in metrics_list if 'ssim_vs_gt' in m]
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # PSNR histogram
    if psnr_values:
        axes[0, 0].hist(psnr_values, bins=20, alpha=0.7, color='blue', edgecolor='black')
        axes[0, 0].set_title('PSNR Distribution')
        axes[0, 0].set_xlabel('PSNR (dB)')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].axvline(np.mean(psnr_values), color='red', linestyle='--', label=f'Mean: {np.mean(psnr_values):.2f}')
        axes[0, 0].legend()
    
    # MSE Improvement histogram
    if improvement_values:
        axes[0, 1].hist(improvement_values, bins=20, alpha=0.7, color='green', edgecolor='black')
        axes[0, 1].set_title('MSE Improvement Distribution')
        axes[0, 1].set_xlabel('Improvement (%)')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].axvline(np.mean(improvement_values), color='red', linestyle='--', label=f'Mean: {np.mean(improvement_values):.1f}%')
        axes[0, 1].legend()
    
    # SSIM histogram
    if ssim_values:
        axes[1, 0].hist(ssim_values, bins=20, alpha=0.7, color='orange', edgecolor='black')
        axes[1, 0].set_title('SSIM Distribution')
        axes[1, 0].set_xlabel('SSIM')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].axvline(np.mean(ssim_values), color='red', linestyle='--', label=f'Mean: {np.mean(ssim_values):.3f}')
        axes[1, 0].legend()
    
    # Correlation plot (PSNR vs Improvement)
    if psnr_values and improvement_values and len(psnr_values) == len(improvement_values):
        axes[1, 1].scatter(psnr_values, improvement_values, alpha=0.6)
        axes[1, 1].set_title('PSNR vs MSE Improvement')
        axes[1, 1].set_xlabel('PSNR (dB)')
        axes[1, 1].set_ylabel('MSE Improvement (%)')
        
        # Add correlation coefficient
        corr = np.corrcoef(psnr_values, improvement_values)[0, 1]
        axes[1, 1].text(0.05, 0.95, f'Correlation: {corr:.3f}', transform=axes[1, 1].transAxes, 
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(Path(output_dir) / "evaluation_metrics.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f" Evaluation plots saved to {Path(output_dir) / 'evaluation_metrics.png'}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained autoencoder model")
    parser.add_argument("model_path", help="Path to trained model checkpoint")
    parser.add_argument("test_dir", help="Directory containing test images")
    parser.add_argument("--output_dir", default="evaluation_results", help="Output directory for results")
    parser.add_argument("--model_type", choices=["basic", "deep"], default="basic", help="Model type")
    parser.add_argument("--device", default="auto", help="Device to use (auto, cuda, cpu)")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model_path):
        print(f" Model file '{args.model_path}' not found!")
        return
    
    if not os.path.exists(args.test_dir):
        print(f" Test directory '{args.test_dir}' not found!")
        return
    
    results = evaluate_model(
        model_path=args.model_path,
        test_dir=args.test_dir,
        output_dir=args.output_dir,
        model_type=args.model_type,
        device=args.device
    )
    
    if results:
        print(f"\n Evaluation complete! Results saved in '{args.output_dir}'")
    else:
        print("\n Evaluation failed!")


if __name__ == "__main__":
    # Example usage if run without arguments
    if len(os.sys.argv) == 1:
        print(" Autoencoder Model Evaluation Tool")
        print("=" * 50)
        print("\nUsage:")
        print("  python evaluate_model.py <model_path> <test_dir> [options]")
        print("\nExample:")
        print("  python evaluate_model.py checkpoints/best_model.pth data/val")
        print("  python evaluate_model.py checkpoints/best_model.pth data/test --model_type deep")
        print("\nThe script will:")
        print("   Load the trained model")
        print("   Process all images in test directory")
        print("   Calculate comprehensive quality metrics")
        print("   Generate comparison visualizations")
        print("   Create evaluation summary and plots")
        print("   Save detailed results as JSON")
    else:
        main()

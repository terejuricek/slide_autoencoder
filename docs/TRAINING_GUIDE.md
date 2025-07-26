# Training Autoencoder Models with Real Histopathology Data

## Complete Step-by-Step Guide

This guide provides detailed instructions for training autoencoder models on real histopathology images using this repository.

## Image Requirements

### **Supported Formats**
- **.png** (recommended for lossless quality)
- **.jpg/.jpeg** (acceptable for most cases)
- **.tiff/.tif** (excellent for high-quality medical images)

### **Image Specifications**

#### **Minimum Requirements:**
- **Size**: At least 256×256 pixels per image
- **Color**: RGB (3-channel) images
- **Bit depth**: 8-bit per channel minimum
- **Quality**: Minimal compression artifacts

#### **Recommended Specifications:**
- **Size**: 512×512 pixels or larger
- **Format**: PNG or TIFF for best quality
- **Resolution**: 20× or 40× magnification preferred
- **Staining**: H&E, IHC, or other standard histology stains
- **Focus**: Well-focused images without motion blur

#### **Data Quality Guidelines:**
```
GOOD Images:
- Clear tissue structures
- Proper staining intensity
- Minimal artifacts
- Good contrast
- Consistent lighting

AVOID Images:
- Severely out of focus
- Over/under-stained
- Heavy compression artifacts
- Extreme lighting variations
- Tissue folds or tears
```

## Step-by-Step Training Process

### **Step 1: Environment Setup**

```bash
# Clone the repository
git clone https://github.com/terejuricek/slide_autoencoder.git
cd slide_autoencoder

# Install dependencies
pip install -r requirements.txt

# Verify GPU availability (recommended)
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### **Step 2: Prepare Your Data Structure**

Create the following directory structure:
```
slide_autoencoder/
├── data/
│   ├── train/          # Training images (70-80% of data)
│   │   ├── image1.png
│   │   ├── image2.tiff
│   │   └── ...
│   ├── val/            # Validation images (20-30% of data)
│   │   ├── val_image1.png
│   │   └── ...
│   └── test/           # Optional test set
│       └── ...
```

**Data Organization Tips:**
```bash
# Create directories
mkdir -p data/train data/val data/test

# Example of organizing your images
# Move 80% to training, 20% to validation
# You can use this script:
```

### **Step 3: Data Preparation Script**

Create a data preparation script:

```python
# prepare_data.py
import os
import shutil
import random
from pathlib import Path

def prepare_histology_data(source_dir, output_dir, train_split=0.8):
    """
    Organize histology images into train/val splits
    
    Args:
        source_dir: Directory containing all your histology images
        output_dir: Output directory (will create train/val subdirs)
        train_split: Fraction for training (0.8 = 80% train, 20% val)
    """
    # Create output directories
    train_dir = Path(output_dir) / "train"
    val_dir = Path(output_dir) / "val"
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all image files
    image_extensions = {'.png', '.jpg', '.jpeg', '.tiff', '.tif'}
    image_files = [f for f in Path(source_dir).glob('*') 
                   if f.suffix.lower() in image_extensions]
    
    print(f"Found {len(image_files)} images")
    
    # Shuffle and split
    random.shuffle(image_files)
    split_idx = int(len(image_files) * train_split)
    
    train_files = image_files[:split_idx]
    val_files = image_files[split_idx:]
    
    # Copy files
    print(f"Copying {len(train_files)} images to training set...")
    for file in train_files:
        shutil.copy2(file, train_dir / file.name)
    
    print(f"Copying {len(val_files)} images to validation set...")
    for file in val_files:
        shutil.copy2(file, val_dir / file.name)
    
    print("Data preparation complete!")
    print(f"Training images: {len(train_files)}")
    print(f"Validation images: {len(val_files)}")

if __name__ == "__main__":
    # Modify these paths for your data
    source_directory = "/path/to/your/histology/images"
    output_directory = "data"
    
    prepare_histology_data(source_directory, output_directory)
```

### **Step 4: Check Your Data**

```python
# check_data.py
import os
from PIL import Image
import numpy as np

def analyze_dataset(data_dir):
    """Analyze your histology dataset"""
    
    image_files = []
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.tif')):
                image_files.append(os.path.join(root, file))
    
    print(f"📊 Dataset Analysis")
    print(f"Total images found: {len(image_files)}")
    
    # Analyze image properties
    sizes = []
    formats = {}
    
    for img_path in image_files[:50]:  # Sample first 50 images
        try:
            with Image.open(img_path) as img:
                sizes.append(img.size)
                fmt = img.format
                formats[fmt] = formats.get(fmt, 0) + 1
        except Exception as e:
            print(f"Error reading {img_path}: {e}")
    
    # Print statistics
    if sizes:
        widths, heights = zip(*sizes)
        print(f"\n📏 Image Dimensions:")
        print(f"Width range: {min(widths)} - {max(widths)} pixels")
        print(f"Height range: {min(heights)} - {max(heights)} pixels")
        print(f"Average size: {np.mean(widths):.0f} × {np.mean(heights):.0f}")
    
    print(f"\n📁 File Formats:")
    for fmt, count in formats.items():
        print(f"{fmt}: {count} files")
    
    # Check minimum requirements
    print(f"\n✅ Quality Check:")
    small_images = sum(1 for w, h in sizes if w < 256 or h < 256)
    if small_images > 0:
        print(f"⚠️  {small_images} images are smaller than 256×256")
    else:
        print("✅ All sampled images meet minimum size requirements")

if __name__ == "__main__":
    analyze_dataset("data")
```

### **Step 5: Model Selection and Configuration**

First, check model requirements:

```python
# Run model analysis
python model_summary.py
```

**Choose your model based on:**

| **Factor** | **Basic Autoencoder** | **Deep Autoencoder** |
|------------|----------------------|---------------------|
| **GPU Memory** | < 4GB | 6GB+ recommended |
| **Image Size** | 256×256 patches | 512×512+ patches |
| **Training Time** | 2-4 hours | 6-10 hours |
| **Quality** | Good for most cases | Superior quality |
| **Dataset Size** | 500+ images | 1000+ images |

### **Step 6: Configure Training Parameters**

Edit `config.py` for your specific needs:

```python
# config.py modifications for real data
TRAINING_CONFIG = {
    # Model selection
    'model_type': 'basic',  # or 'deep' for higher quality
    
    # Data parameters
    'patch_size': 256,      # Adjust based on your image size
    'batch_size': 8,        # Reduce if out of memory
    'num_workers': 4,       # Adjust based on your CPU
    
    # Training parameters
    'num_epochs': 100,      # Increase for larger datasets
    'learning_rate': 1e-4,  # Standard starting point
    'weight_decay': 1e-5,   # Regularization
    
    # Data augmentation (tune for your data)
    'noise_factor': 0.3,    # Reduce for high-quality images
    'augmentation_prob': 0.5,  # Probability of applying augmentations
    
    # Validation
    'val_frequency': 5,     # Validate every N epochs
    'save_frequency': 10,   # Save checkpoint every N epochs
}
```

### **Step 7: Start Training**

**Option A: Simple Training (Recommended)**
```python
from train import train_autoencoder

# Train with your data
trainer, history = train_autoencoder(
    data_dir="data/train",
    val_dir="data/val",      # Optional validation directory
    model_type="basic",      # or "deep"
    batch_size=8,           # Adjust based on GPU memory
    num_epochs=100,
    learning_rate=1e-4,
    patch_size=256,
    device="cuda" if torch.cuda.is_available() else "cpu"
)
```

**Option B: Advanced Training with Custom Parameters**
```python
from train import HistologyTrainer
from models import HistoAutoencoder
from data_utils import get_data_loaders
import torch

# Create data loaders
train_loader, val_loader = get_data_loaders(
    train_dir="data/train",
    val_dir="data/val",
    batch_size=8,
    patch_size=256,
    num_workers=4
)

# Initialize model and trainer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = HistoAutoencoder(input_channels=3, base_channels=64)
trainer = HistologyTrainer(model, device, learning_rate=1e-4)

# Train the model
history = trainer.train(
    train_loader=train_loader,
    val_loader=val_loader,
    num_epochs=100,
    save_dir="checkpoints"
)
```

### **Step 8: Monitor Training Progress**

During training, monitor these key metrics:

```bash
# Training will display:
Epoch 1/100: Train Loss: 0.0234, Val Loss: 0.0198, LR: 1e-4
Epoch 2/100: Train Loss: 0.0187, Val Loss: 0.0165, LR: 1e-4
...

# Look for:
- Decreasing training loss
- Validation loss following training loss
- No sudden spikes in loss
-!  Watch for overfitting (val loss increasing while train loss decreases)
```

**Training visualization files generated:**
- `training_history.png` - Loss curves
- `sample_reconstructions.png` - Before/after examples
- `checkpoints/best_model.pth` - Best model weights

### **Step 9: Evaluate Results**

```python
# evaluate_model.py
from inference import HistologyDenoiser
import cv2
import numpy as np

# Load trained model
denoiser = HistologyDenoiser("checkpoints/best_model.pth", model_type="basic")

# Test on validation images
test_dir = "data/val"
output_dir = "results/validation_results"

# Batch process validation set
metrics = denoiser.batch_denoise_directory(
    input_dir=test_dir,
    output_dir=output_dir,
    calculate_metrics=True
)

print(f"Average PSNR improvement: {metrics['avg_psnr']:.2f} dB")
print(f"Average MSE reduction: {metrics['avg_mse']:.4f}")
```

### **Step 10: Fine-tune if Needed**

**If results are not satisfactory:**

1. **Increase training epochs**: Try 200-500 epochs
2. **Adjust learning rate**: Try 5e-5 or 2e-4
3. **Modify noise factor**: Reduce to 0.1-0.2 for cleaner images
4. **Use deep model**: Switch to `DeepHistoAutoencoder`
5. **Add more data**: Collect additional training images

**Advanced fine-tuning:**
```python
# Resume training from checkpoint
trainer.load_checkpoint("checkpoints/best_model.pth")
trainer.learning_rate = 5e-5  # Lower learning rate for fine-tuning

# Train for additional epochs
additional_history = trainer.train(
    train_loader=train_loader,
    val_loader=val_loader,
    num_epochs=50,  # Additional epochs
    save_dir="checkpoints"
)
```

## Training Tips and Best Practices

### **Data Quality Tips:**
```
1. Pre-filter images: Remove heavily artifact-laden images
2. Consistent staining: Group by staining protocol if possible
3. Magnification consistency: Use similar magnification levels
4. Color normalization: Consider color standardization for multi-center data
```

### **Training Optimization:**
```
1. Start small: Train with 100-200 images first to test setup
2. Monitor GPU usage: Use nvidia-smi to check memory utilization
3. Save frequently: Enable checkpoint saving every 10 epochs
4. Validate regularly: Check validation loss every 5 epochs
```

### **Hardware Recommendations:**
```
Minimum Setup:
- GPU: 4GB VRAM (GTX 1660, RTX 3060)
- RAM: 16GB system memory
- Storage: 50GB+ available space

Recommended Setup:
- GPU: 8GB+ VRAM (RTX 3070, RTX 4070)
- RAM: 32GB system memory
- Storage: 100GB+ SSD storage
```

### **Expected Training Times:**
```
Basic Model (256×256, 1000 images):
- GPU (RTX 3070): 2-3 hours
- GPU (GTX 1660): 4-6 hours
- CPU only: 20-30 hours (not recommended)

Deep Model (256×256, 1000 images):
- GPU (RTX 3070): 6-8 hours
- GPU (GTX 1660): 12-16 hours
```

## Troubleshooting Common Issues

### **Memory Errors:**
```bash
# Reduce batch size
batch_size = 4  # Instead of 8

# Reduce patch size
patch_size = 224  # Instead of 256

# Use gradient checkpointing (advanced)
model.gradient_checkpointing = True
```

### **Poor Results:**
```python
# Increase model capacity
model = HistoAutoencoder(base_channels=128)  # Instead of 64

# Adjust loss weights
reconstruction_loss = 0.8 * mse + 0.2 * l1  # More weight on MSE

# Increase training time
num_epochs = 200  # Instead of 100
```

### **Training Instability:**
```python
# Add gradient clipping
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# Reduce learning rate
learning_rate = 5e-5  # Instead of 1e-4

# Add weight decay
weight_decay = 1e-4  # Instead of 1e-5
```

## Final Checklist

Before starting training, ensure:

- [ ] Data is properly organized in train/val directories
- [ ] Images meet minimum size requirements (256×256+)
- [ ] GPU memory is sufficient for chosen batch size
- [ ] Validation set is representative of training data
- [ ] Training parameters are appropriate for dataset size
- [ ] Adequate storage space for checkpoints and results
- [ ] Monitoring tools are set up for training progress

**Ready to train? Run:**
```bash
python train.py
```

Or use the interactive demo:
```bash
python demo_visualization.py  # First understand the models
python model_summary.py       # Check requirements
python train.py              # Start training
```

This comprehensive guide ensures successful training of autoencoder models on your real histopathology data!

# Histopathology Autoencoder for Image Denoising

This project implements autoencoder architectures specifically designed for denoising histopathological images. The models are optimized for preserving crucial diagnostic details while removing various types of noise commonly found in digital pathology.

## Quick Setup

### Easy Setup (Recommended)
Run the automated setup script to create a virtual environment and install all dependencies:

```bash
./setup_environment.sh
```

This will:
1. Create a virtual environment named `slide_env`
2. Install all required dependencies
3. Set up the project for immediate use

### Manual Setup
If you prefer manual setup:

```bash
# Create virtual environment
python -m venv slide_env

# Activate environment
source slide_env/bin/activate  # On macOS/Linux
# OR
slide_env\Scripts\activate     # On Windows

# Install dependencies
pip install -r requirements.txt

# Test setup
python scripts/quick_start_real_data.py
```

### Important Note
Due to NumPy 2.x compatibility issues with some packages, this project uses a virtual environment with NumPy < 2.0.0. Always activate the virtual environment before running any scripts:

```bash
source slide_env/bin/activate
```

## Getting Training Data

### Option 1: Quick Start with Public Datasets (Recommended)

The easiest way to get started is using our automated dataset downloader:

```bash
# Quick setup with a small test dataset (~150MB)
./quick_start_dataset.sh

# Or download specific datasets manually
python scripts/download_datasets.py --dataset breast-cancer
python scripts/download_datasets.py --dataset colorectal --prepare
```

**Available datasets:**
- `breast-cancer`: Breast cancer histopathology (~1.3GB, 277K patches)
- `patchcamelyon`: Lymph node metastasis detection (~7GB, 327K patches)  
- `colorectal`: Colorectal tissue classification (~150MB, 5K images)
- `openslide`: Sample WSI files for testing

### Option 2: Use Your Own Images

If you have your own histopathology images:

```bash
# Prepare your data
python -m src.training.prepare_data /path/to/your/images

# This will create data/train/ and data/val/ directories
```

**Image requirements:**
- **Formats**: PNG, JPEG, TIFF
- **Size**: Minimum 256×256 pixels
- **Type**: RGB histopathology images
- **Quantity**: 100+ for testing, 500+ for good results

For detailed information about public datasets, see [`docs/PUBLIC_DATASETS.md`](docs/PUBLIC_DATASETS.md).

## Features

- **Specialized Architectures**: Two autoencoder variants optimized for histology images
- **Robust Noise Handling**: Handles Gaussian noise, salt & pepper noise, blur, and scanner artifacts
- **Skip Connections**: Preserves fine details crucial for pathological diagnosis
- **Patch-based Processing**: Efficient processing of large whole slide images (WSI)
- **Comprehensive Training Pipeline**: Complete training, validation, and evaluation framework
- **Flexible Inference**: Single image or batch processing capabilities

## Model Architectures

Project implements **two types of convolutional autoencoders**, both specifically designed for histopathological image denoising:

### 1. Basic Autoencoder (`HistoAutoencoder`)

**Type**: **U-Net Style Convolutional Autoencoder with Skip Connections**
- **Parameters**: 34.5 million
- **Architecture**: Symmetric encoder-decoder with U-Net skip connections
- **Design Pattern**: Feature pyramid with progressive channel expansion

#### **Detailed Structure:**

##### ENCODER (Downsampling Path):
```
Input (3, 256, 256)
├── Encoder Block 1: 3 → 64 channels, MaxPool → (64, 128, 128)
│   ├── Conv2D(3→64, 3×3) + BatchNorm + LeakyReLU
│   ├── Conv2D(64→64, 3×3) + BatchNorm + LeakyReLU
│   └── MaxPool2D(2×2)
├── Encoder Block 2: 64 → 128 channels → (128, 64, 64)
├── Encoder Block 3: 128 → 256 channels → (256, 32, 32)
└── Encoder Block 4: 256 → 512 channels → (512, 16, 16)
```

##### BOTTLENECK (Feature Compression):
```
Input: (512, 16, 16)
├── Conv2D(512→1024, 3×3) + BatchNorm + LeakyReLU
└── Conv2D(1024→1024, 3×3) + BatchNorm + LeakyReLU
Output: (1024, 16, 16)
```

##### DECODER (Upsampling with Skip Connections):
```
Bottleneck: (1024, 16, 16)
├── Decoder4: ConvTranspose2D → (512, 32, 32)
│   └── Skip connect with Encoder3 → (512+256, 32, 32)
├── Decoder3: Process → (256, 64, 64)
│   └── Skip connect with Encoder2 → (256+128, 64, 64)
├── Decoder2: Process → (128, 128, 128)
│   └── Skip connect with Encoder1 → (128+64, 128, 128)
├── Decoder1: Process → (64, 256, 256)
└── Final Layer: Conv2D → (3, 256, 256) + Sigmoid
```

### 2. Deep Autoencoder (`DeepHistoAutoencoder`)

**Type**: **ResNet-Style Autoencoder with Residual Blocks**
- **Parameters**: 133.3 million
- **Architecture**: Deep residual network with U-Net skip connections
- **Design Pattern**: Residual learning + hierarchical feature extraction

#### **Key Architectural Features:**

##### RESIDUAL BLOCKS:
```python
class ResidualBlock:
    Input → Conv2D → BatchNorm → LeakyReLU → Conv2D → BatchNorm → (+Input) → LeakyReLU
    |___________________________________________________________________|
                              Skip Connection
```

##### ENHANCED ENCODER:
```
Input (3, 256, 256)
├── Initial Conv: 7×7 Conv → (64, 256, 256)
├── Encoder1: Downsample + 2×ResidualBlocks → (128, 128, 128)
├── Encoder2: Downsample + 2×ResidualBlocks → (256, 64, 64)
├── Encoder3: Downsample + 2×ResidualBlocks → (512, 32, 32)
└── Encoder4: Downsample + 2×ResidualBlocks → (1024, 16, 16)
```

##### RESIDUAL BOTTLENECK:
```
Input: (1024, 16, 16)
├── ResidualBlock(1024) ×3
└── Deep feature processing while maintaining dimensions
```

##### RESIDUAL DECODER:
```
Each decoder layer includes:
├── ConvTranspose2D (2× upsampling)
├── Skip connection concatenation
└── ResidualBlock for feature refinement
```

## Architectural Classification

### **Primary Type: Convolutional Autoencoders**

| **Aspect** | **Basic Model** | **Deep Model** |
|------------|-----------------|----------------|
| **Architecture Family** | U-Net Autoencoder | ResNet + U-Net Hybrid |
| **Skip Connections** | Feature concatenation | Feature concatenation + Residuals |
| **Depth** | 4 encoder/decoder levels | 4 levels + residual blocks |
| **Complexity** | Moderate (34.5M params) | High (133.3M params) |
| **Training Speed** | Fast | Slower but more stable |
| **Quality** | Good for most cases | Superior for complex patterns |

### **Key Design Patterns:**

#### **1. U-Net Architecture (Both Models)**
```
Encoder: Progressive downsampling with feature expansion
Decoder: Progressive upsampling with skip connections
Skip Connections: Preserve spatial details at each resolution
```

#### **2. Residual Learning (Deep Model Only)**
```
Identity Mapping: f(x) = F(x) + x
Purpose: Enables deeper networks without vanishing gradients
Benefit: Better feature learning and gradient flow
```

#### **3. Multi-Scale Feature Processing**
```
Level 1: (256×256) - Fine details and textures
Level 2: (128×128) - Cellular structures  
Level 3: (64×64)   - Tissue patterns
Level 4: (32×32)   - High-level features
Bottleneck: (16×16) - Compressed representation
```

## Histopathology-Specific Optimizations

### **Medical Image Adaptations:**
- **LeakyReLU Activation**: Prevents dead neurons common in medical imaging
- **Batch Normalization**: Stabilizes training with varying staining intensities
- **Sigmoid Output**: Ensures pixel values in [0,1] range for medical images
- **Skip Connections**: Preserves diagnostic details crucial for pathological analysis

### **Noise Handling Capabilities:**
- **Multi-scale Processing**: Handles noise at different frequency levels
- **Residual Learning**: Learns to remove noise while preserving signal
- **Progressive Feature Extraction**: Separates noise from tissue patterns
- **Deep Feature Representation**: Robust to various noise types

### **Channel Progression Strategy:**
```
Input Channels:    3 (RGB histology image)
Encoder Channels:  3 → 64 → 128 → 256 → 512 → 1024
Decoder Channels:  1024 → 512 → 256 → 128 → 64 → 3
Skip Connections:  Concatenate features at matching resolutions
```

### **Mathematical Formulation:**

- **Encoder Function**: `E: ℝ^(256×256×3) → ℝ^(16×16×1024)`
- **Decoder Function**: `D: ℝ^(16×16×1024) → ℝ^(256×256×3)`
- **Skip Function**: `S_i: ℝ^(H/2^i×W/2^i×C_i) → ℝ^(H/2^(4-i)×W/2^(4-i)×C_i)`

**Complete Autoencoder**: `f(x) = D(E(x) ⊕ S(E(x)))`
where `⊕` represents concatenation at matching spatial resolutions.

- 4-level encoder-decoder with skip connections
- Optimized for 256x256 patches
- **34.5 million parameters** (corrected)
- Suitable for real-time applications

### 2. Deep Autoencoder (`DeepHistoAutoencoder`)
- Residual blocks for complex pattern recognition
- Better for high-resolution images (512x512+)
- **133.3 million parameters** (corrected)
- Suitable for high-quality denoising

## Model Visualization and Analysis

The project includes comprehensive visualization tools to understand model architectures, data flow, and performance characteristics.

### Available Visualization Tools

#### 1. **Interactive Visualization System** (`model_visualization.py`)
Provides detailed analysis of both autoencoder models:

```bash
python -m src.visualization.model_visualization
```

**Features:**
- **Architecture Maps**: Layer-by-layer breakdown with input/output shapes
- **Parameter Analysis**: Distribution of parameters across components
- **Memory Usage**: Training and inference memory requirements
- **Data Flow Visualization**: Visual representation of tensor flow
- **Model Comparison**: Side-by-side comparison of both architectures

#### 2. **Quick Demo** (`demo_visualization.py`)
Fast overview of key model characteristics:

```bash
python -m src.visualization.demo_visualization
```

#### 3. **Integrated Overview** (`test/overview.py`)
Complete system overview including visualizations:

```bash
python tests/overview.py
```

### Model Architecture Maps

#### Basic Autoencoder Flow:
```
Input [B,3,256,256] → Encoder → Bottleneck [B,1024,16,16] → Decoder → Output [B,3,256,256]
                          ↓         ↑              ↓         ↑
                    Skip Connections preserve spatial details
```

#### Parameter Distribution:
- **Encoder**: ~40% of parameters (feature extraction)
- **Bottleneck**: ~35% of parameters (feature compression)
- **Decoder**: ~20% of parameters (reconstruction)
- **Final Layer**: ~5% of parameters (output refinement)

#### Memory Requirements (Batch Size 8):
- **Basic Model**: ~2.5 GB GPU memory for training
- **Deep Model**: ~6.8 GB GPU memory for training
- **Inference**: 60-70% less memory than training

### Visualization Output Examples

The visualization system provides detailed insights such as:

```
ARCHITECTURE MAP - BASIC AUTOENCODER
====================================
Model Type: HistoAutoencoder
Total Parameters: 34,544,963
Model Size: 131.78 MB
Estimated Forward Pass Memory: 423.00 MB

LAYER-BY-LAYER BREAKDOWN:
Layer Name               Type        Input Shape         Output Shape      Params
encoder1.0              Conv2d      (1,3,256,256)      (1,64,256,256)    1,792
encoder1.1              BatchNorm2d (1,64,256,256)     (1,64,256,256)    128
...
```

**Usage Tips:**
- Use `model_comparison` to choose between Basic vs Deep models
- Check `memory_analysis` before training to ensure adequate GPU memory
- Review `parameter_breakdown` to understand model complexity
- Use `data_flow_map` to visualize skip connections and tensor shapes

## Installation

```bash
# Clone the repository
git clone https://github.com/terejuricek/slide_autoencoder.git
cd slide_autoencoder

# Install in development mode
pip install -e .

# Or install dependencies directly
pip install -r requirements.txt
```

## Quick Start

### 1. Interactive Quick Start

```bash
# Run the interactive setup guide
python scripts/quick_start_real_data.py
```

### 2. Prepare Your Data

```bash
# Prepare real histopathology data
python -m src.training.prepare_data /path/to/your/images
```

### 3. Train a Model

```bash
# Quick training with basic model
python -m src.training.train_real_data data/train

# Advanced training with deep model
python -m src.training.train_real_data data/train --model_type deep --batch_size 4
```

### 4. Visualize Model Architecture

```bash
# Interactive model visualization
python -m src.visualization.model_visualization
```

### 5. Evaluate Results

```bash
# Comprehensive model evaluation
python -m src.evaluation.evaluate_model checkpoints/best_model.pth data/val
```

### 6. Denoise Images

```python
from src.slide_autoencoder.inference import HistologyDenoiser

# Load trained model and denoise images
denoiser = HistologyDenoiser("checkpoints/best_model.pth")
denoiser.batch_denoise_directory("input_images/", "denoised_output/")
```

## Project Structure

```
slide_autoencoder/
├── README.md                           # Project overview and documentation
├── CHANGELOG.md                        # Version history
├── setup.py                            # Package installation script
├── pyproject.toml                      # Python packaging configuration
├── Makefile                            # Development automation
├── requirements.txt                    # Core dependencies
├── src/                                # Source code
│   ├── slide_autoencoder/              # Main package
│   │   ├── __init__.py                 # Package initialization
│   │   ├── models.py                   # Autoencoder architectures
│   │   ├── data_utils.py               # Data loading and preprocessing
│   │   └── inference.py                # Inference and evaluation
│   ├── training/                       # Training modules
│   │   ├── __init__.py
│   │   ├── train.py                    # Basic training pipeline
│   │   ├── train_real_data.py          # Real data training
│   │   └── prepare_data.py             # Data preparation utilities
│   ├── evaluation/                     # Evaluation modules
│   │   ├── __init__.py
│   │   └── evaluate_model.py           # Model evaluation and metrics
│   └── visualization/                  # Visualization modules
│       ├── __init__.py
│       ├── model_visualization.py      # Interactive model analysis
│       ├── demo_visualization.py       # Quick demonstrations
│       └── model_summary.py            # Model comparison tools
├── scripts/                            # Utility scripts
│   ├── quick_start_real_data.py        # Quick start guide
│   ├── demo.py                         # Demonstration script
│   └── debug_model.py                  # Debugging utilities
├── tests/                              # Test suite
│   └── overview.py                     # System overview tests
├── docs/                               # Documentation
│   ├── index.md                        # Documentation index
│   ├── TRAINING_GUIDE.md               # Complete training guide
│   ├── REAL_DATA_TRAINING_STEPS.md     # Step-by-step instructions
│   ├── TRAINING_CHECKLIST.md           # Quality assurance checklist
│   └── SOLUTION_SUMMARY.md             # Project overview
├── configs/                            # Configuration files
│   └── config.py                       # Training configurations
├── examples/                           # Example images and outputs
├── data/                               # Training data directory (created during use)
├── checkpoints/                        # Saved models (created during training)
└── results/                            # Inference results (created during use)
```

## Key Features for Histopathology

### Noise Types Handled
- **Gaussian Noise**: Random pixel variations
- **Salt & Pepper**: Random white/black pixels
- **Motion Blur**: Camera shake or scanning artifacts
- **Defocus Blur**: Focus-related issues
- **Compression Artifacts**: JPEG compression noise

### Histology-Specific Optimizations
- **Color Preservation**: Maintains H&E and IHC staining characteristics
- **Cellular Detail**: Preserves nuclear and cytoplasmic features
- **Texture Retention**: Maintains tissue architecture patterns
- **Multi-scale Processing**: Handles various magnification levels

## Training Details

### Loss Function
The model uses a combined loss function optimized for histopathological images:
```
Total Loss = 0.7 * MSE + 0.3 * L1 + 0.1 * Perceptual Loss
```

- **MSE**: Overall reconstruction quality
- **L1**: Sharp edge preservation
- **Perceptual**: Texture and structural information

### Data Augmentation
- Random horizontal/vertical flips
- Random 90-degree rotations
- Color jittering (brightness, contrast, saturation)
- Dynamic noise injection during training

### Training Features
- Automatic best model saving
- Learning rate scheduling
- Gradient clipping
- Training history visualization
- Comprehensive evaluation metrics

## Usage Examples

### Custom Model Training
```python
from src.slide_autoencoder.models import HistoAutoencoder
from src.training.train import HistologyTrainer

# Create custom model
model = HistoAutoencoder(input_channels=3, base_channels=64)
trainer = HistologyTrainer(model, device)

# Train with custom parameters
history = trainer.train(train_loader, val_loader, num_epochs=200)
```

### Advanced Inference
```python
from src.slide_autoencoder.inference import HistologyDenoiser
import cv2

# Load and configure denoiser
denoiser = HistologyDenoiser("checkpoints/best_model.pth", model_type="deep")

# Process large WSI with overlapping patches
large_image = cv2.imread("whole_slide_image.tif")
denoised = denoiser.denoise_image(large_image, patch_size=512, overlap=64)

# Calculate quality metrics
metrics = denoiser.calculate_metrics(original, denoised, ground_truth)
print(f"PSNR: {metrics['psnr_vs_gt']:.2f} dB")
```

## Performance Metrics

The models are evaluated using:
- **PSNR (Peak Signal-to-Noise Ratio)**: Quantitative quality measure
- **MSE (Mean Squared Error)**: Pixel-level accuracy
- **Visual Quality Assessment**: Preservation of diagnostic features
- **Processing Speed**: Inference time per patch

### Model Performance Comparison

| **Metric** | **Basic Autoencoder** | **Deep Autoencoder** |
|------------|----------------------|---------------------|
| **Parameters** | 34.5M | 133.3M |
| **Training Time** | 2-4 hours (1000 patches) | 6-10 hours (1000 patches) |
| **Inference Speed** | ~50ms per 256×256 patch | ~120ms per 256×256 patch |
| **GPU Memory** | ~2GB (batch size 16) | ~4GB (batch size 8) |
| **Use Case** | Real-time applications | High-quality research |
| **Quality** | Good for most cases | Superior reconstruction |

## Requirements

- Python 3.7+
- PyTorch 1.9+
- torchvision 0.10+
- OpenCV 4.5+
- NumPy, Matplotlib, Pillow
- tqdm (for progress bars)

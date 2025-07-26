#!/bin/bash
# Quick Start: Download and Prepare a Test Dataset

echo "🔬 Quick Start: Histopathology Dataset Setup"
echo "=============================================="
echo "This script will download a small test dataset and prepare it for training."
echo ""

# Activate environment
echo "📦 Activating environment..."
source slide_env/bin/activate

# Check if Kaggle is configured
if [ ! -f ~/.kaggle/kaggle.json ]; then
    echo ""
    echo "🔑 Kaggle API Setup Required"
    echo "=============================="
    echo "To download datasets, you need Kaggle credentials:"
    echo ""
    echo "1. Go to: https://www.kaggle.com/account"
    echo "2. Click 'Create New API Token'"
    echo "3. Download kaggle.json"
    echo "4. Run these commands:"
    echo "   mkdir -p ~/.kaggle"
    echo "   cp ~/Downloads/kaggle.json ~/.kaggle/"
    echo "   chmod 600 ~/.kaggle/kaggle.json"
    echo ""
    echo "After setting up credentials, run this script again."
    exit 1
fi

echo "✅ Kaggle credentials found!"
echo ""

# Download a small dataset (colorectal is smallest)
echo "📥 Downloading Colorectal Histology Dataset (~150MB)..."
echo "This is a good starter dataset with 5,000 images in 8 tissue classes."
echo ""

python scripts/download_datasets.py --dataset colorectal --prepare

if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 SUCCESS! Dataset downloaded and prepared."
    echo "=============================================="
    echo ""
    echo "📁 Your data is ready in: prepared_data_1/"
    echo ""
    echo "🚀 Next steps:"
    echo "1. Start training:"
    echo "   python -m src.training.train_real_data prepared_data_1/train"
    echo ""
    echo "2. Monitor training progress"
    echo "3. Evaluate results:"
    echo "   python -m src.evaluation.evaluate_model checkpoints/best_model.pth prepared_data_1/val"
    echo ""
    echo "⚙️ Training tips:"
    echo "• Start with basic model for faster training"
    echo "• Use smaller patch size (256) for this dataset"
    echo "• Expect 1-2 hours training time on GPU"
    echo ""
else
    echo ""
    echo "❌ Download failed. Please check:"
    echo "• Internet connection"
    echo "• Kaggle credentials"
    echo "• Available disk space (~200MB needed)"
fi

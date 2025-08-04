#!/bin/bash
# Install optional dependencies for dataset downloading

echo "Installing optional dependencies for dataset downloading..."

source slide_env/bin/activate

pip install --no-user requests>=2.25.0 kaggle>=1.5.12

echo "Optional dependencies installed!"
echo ""
echo "Now you can download datasets with:"
echo "  python scripts/download_datasets.py --dataset breast-cancer"
echo "  python scripts/download_datasets.py --dataset patchcamelyon" 
echo "  python scripts/download_datasets.py --dataset colorectal"
echo ""
echo "Note: For Kaggle datasets, you'll need to set up API credentials"
echo "Visit: https://www.kaggle.com/account → Create New API Token"

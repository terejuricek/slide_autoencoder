#!/bin/bash
# Simple activation script for the slide autoencoder environment

echo "Activating Slide Autoencoder environment..."
source slide_env/bin/activate

echo "Environment activated! You can now run:"
echo "  python scripts/quick_start_real_data.py"
echo "  python -m src.training.train_real_data data/train"
echo ""
echo "To deactivate, type: deactivate"

# Start a new shell with the environment activated
exec $SHELL

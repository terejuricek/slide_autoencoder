#!/bin/bash
# Slide Autoencoder Environment Setup Script

echo "Setting up Slide Autoencoder environment..."

# Create virtual environment if it doesn't exist
if [ ! -d "slide_env" ]; then
    echo "Creating virtual environment..."
    python -m venv slide_env
fi

# Activate virtual environment
echo "Activating virtual environment..."
source slide_env/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install --no-user -r requirements.txt

echo "Environment setup complete!"
echo "To activate the environment in the future, run:"
echo "source slide_env/bin/activate"
echo ""
echo "To test the setup, run:"
echo "python scripts/quick_start_real_data.py"

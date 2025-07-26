"""
Slide Autoencoder - Histopathology Image Denoising

This package provides autoencoder architectures specifically designed for denoising 
histopathological images while preserving diagnostic details.
"""

from .models import HistoAutoencoder, DeepHistoAutoencoder
from .data_utils import HistologyDataset, create_synthetic_patches
from .inference import HistologyDenoiser

__version__ = "1.0.0"
__author__ = "Tereza Jurickova"

__all__ = [
    "HistoAutoencoder",
    "DeepHistoAutoencoder", 
    "HistologyDataset",
    "create_synthetic_patches",
    "HistologyDenoiser"
]

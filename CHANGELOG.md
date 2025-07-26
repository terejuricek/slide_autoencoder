# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2025-07-22

### Added
- Initial development release of slide_autoencoder package
- HistoAutoencoder: U-Net style autoencoder with skip connections (34.5M parameters)
- DeepHistoAutoencoder: ResNet-style autoencoder with residual blocks (133.3M parameters)
- Comprehensive training pipeline for real histopathology data
- Data preparation and quality checking utilities
- Model evaluation and metrics calculation
- Interactive visualization tools for model architectures
- Complete documentation and training guides
- Professional repository structure
- Support for PNG, JPEG, and TIFF image formats
- Batch processing capabilities for large datasets
- Memory-efficient patch-based processing
- Comprehensive testing suite

### Features
- Specialized noise handling for histopathology images
- Skip connections for preserving diagnostic details
- Multi-scale feature processing
- Automated data preparation with quality checks
- Real-time training monitoring and visualization
- Comprehensive evaluation metrics (PSNR, SSIM, MSE)
- Interactive model comparison and selection tools
- Memory usage analysis and optimization
- Professional packaging with setuptools and pyproject.toml

### Documentation
- Complete README with architecture details
- Step-by-step training guide
- Training checklist for quality assurance
- API documentation with examples
- Installation and setup instructions

### Repository Structure
- Organized into logical folders following Python best practices
- Separate modules for training, evaluation, and visualization
- Professional packaging configuration
- Comprehensive test coverage
- Development and documentation requirements

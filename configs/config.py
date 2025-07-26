# Configuration file for histopathology autoencoder

# Model Configuration
MODEL_CONFIG = {
    "basic": {
        "input_channels": 3,
        "base_channels": 64,
        "description": "Lightweight autoencoder for fast inference"
    },
    "deep": {
        "input_channels": 3,
        "base_channels": 64,
        "description": "Deep autoencoder with residual blocks for high quality"
    }
}

# Training Configuration
TRAINING_CONFIG = {
    "learning_rate": 1e-4,
    "weight_decay": 1e-5,
    "batch_size": 16,
    "num_epochs": 100,
    "patch_size": 256,
    "noise_factor": 0.3,
    "validation_split": 0.2,
    "num_workers": 4,
    "save_every": 10
}

# Loss Function Weights
LOSS_WEIGHTS = {
    "mse_weight": 0.7,
    "l1_weight": 0.3,
    "perceptual_weight": 0.1
}

# Data Augmentation Configuration
AUGMENTATION_CONFIG = {
    "horizontal_flip_prob": 0.5,
    "vertical_flip_prob": 0.5,
    "rotation_degrees": 90,
    "color_jitter": {
        "brightness": 0.2,
        "contrast": 0.2,
        "saturation": 0.2,
        "hue": 0.1
    },
    "normalize_mean": [0.485, 0.456, 0.406],  # ImageNet normalization
    "normalize_std": [0.229, 0.224, 0.225]
}

# Noise Simulation Configuration
NOISE_CONFIG = {
    "gaussian_prob": 0.5,
    "gaussian_std": 0.1,
    "salt_pepper_prob": 0.3,
    "salt_pepper_amount": 0.02,
    "blur_prob": 0.3,
    "blur_kernel_sizes": [3, 5, 7],
    "motion_blur_prob": 0.5  # probability of motion blur vs gaussian blur
}

# Inference Configuration
INFERENCE_CONFIG = {
    "default_patch_size": 256,
    "default_overlap": 32,
    "batch_processing_size": 4,
    "supported_formats": ['.png', '.jpg', '.jpeg', '.tiff', '.tif', '.bmp']
}

# Paths Configuration
PATHS = {
    "data_dir": "data",
    "train_dir": "data/train",
    "val_dir": "data/val",
    "test_dir": "data/test",
    "synthetic_dir": "data/synthetic",
    "checkpoints_dir": "checkpoints",
    "results_dir": "results",
    "logs_dir": "logs"
}

# Performance Monitoring
METRICS = {
    "primary_metric": "val_loss",
    "monitor_metrics": ["train_loss", "val_loss", "psnr", "mse"],
    "early_stopping_patience": 15,
    "lr_scheduler_patience": 10,
    "lr_scheduler_factor": 0.5
}

# Hardware Configuration
HARDWARE = {
    "use_cuda": True,
    "cuda_device": 0,
    "mixed_precision": False,  # Use automatic mixed precision for faster training
    "pin_memory": True,
    "non_blocking": True
}

# Histopathology Specific Settings
HISTOPATHOLOGY = {
    "stain_types": ["HE", "IHC", "special"],
    "magnifications": ["4x", "10x", "20x", "40x"],
    "tissue_types": ["epithelial", "connective", "muscle", "nervous"],
    "preserve_colors": True,
    "preserve_textures": True,
    "diagnostic_quality": True
}

# Logging Configuration
LOGGING = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "log_to_file": True,
    "log_training_images": True,
    "tensorboard_logging": True
}

# Validation and Testing
VALIDATION = {
    "metrics_to_track": ["mse", "psnr", "ssim"],
    "save_validation_images": True,
    "validation_frequency": 5,  # epochs
    "test_time_augmentation": False
}

# Export settings for easy access
__all__ = [
    'MODEL_CONFIG',
    'TRAINING_CONFIG', 
    'LOSS_WEIGHTS',
    'AUGMENTATION_CONFIG',
    'NOISE_CONFIG',
    'INFERENCE_CONFIG',
    'PATHS',
    'METRICS',
    'HARDWARE',
    'HISTOPATHOLOGY',
    'LOGGING',
    'VALIDATION'
]

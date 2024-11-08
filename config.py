"""Configuration settings for the ISL recognition system"""

# Dataset Configuration
DATASET_CONFIG = {
    'IMAGE_SIZE': (224, 224),
    'ROI_SIZE': 300,
    'BATCH_SIZE': 32,  # Optimized for 4GB VRAM
    'VALIDATION_SPLIT': 0.15,
    'TEST_SPLIT': 0.15,
    'MIN_IMAGES_PER_CLASS': 100,
    'RECOMMENDED_IMAGES_PER_CLASS': 150
}

# Training Configuration
TRAINING_CONFIG = {
    'EPOCHS': 50,
    'INITIAL_LEARNING_RATE': 0.001,
    'MIN_LEARNING_RATE': 1e-6,
    'DECAY_STEPS': 1000,
    'DECAY_RATE': 0.9,
    'EARLY_STOPPING_PATIENCE': 5,
    'REDUCE_LR_PATIENCE': 3,
    'REDUCE_LR_FACTOR': 0.2
}

# Model Configuration
MODEL_CONFIG = {
    'FINE_TUNE_LAYERS': 20,
    'DENSE_LAYERS': [256, 128],
    'DROPOUT_RATES': [0.5, 0.3]
}

# Data Augmentation Configuration
AUGMENTATION_CONFIG = {
    'ROTATION_RANGE': 0.1,
    'ZOOM_RANGE': 0.1,
    'BRIGHTNESS_RANGE': 0.2,
    'CONTRAST_RANGE': 0.2
}

# System Configuration
SYSTEM_CONFIG = {
    'GPU_MEMORY_LIMIT': 3072,  # 3GB (leaving 1GB for system)
    'MIXED_PRECISION': True,   # Enable mixed precision for better memory usage
    'NUM_PARALLEL_CALLS': 4    # For data loading
}
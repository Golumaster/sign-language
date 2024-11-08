import tensorflow as tf
from tensorflow.keras import layers, models, applications
from config import MODEL_CONFIG, TRAINING_CONFIG, DATASET_CONFIG, AUGMENTATION_CONFIG

def create_model(num_classes):
    """Create and return the model architecture"""
    base_model = applications.MobileNetV2(
        input_shape=(*DATASET_CONFIG['IMAGE_SIZE'], 3),
        include_top=False,
        weights='imagenet'
    )
    
    # Freeze early layers
    for layer in base_model.layers[:-MODEL_CONFIG['FINE_TUNE_LAYERS']]:
        layer.trainable = False
    
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.BatchNormalization(),
        layers.Dense(MODEL_CONFIG['DENSE_LAYERS'][0], activation='relu'),
        layers.Dropout(MODEL_CONFIG['DROPOUT_RATES'][0]),
        layers.Dense(MODEL_CONFIG['DENSE_LAYERS'][1], activation='relu'),
        layers.Dropout(MODEL_CONFIG['DROPOUT_RATES'][1]),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

def create_data_augmentation():
    """Create data augmentation pipeline"""
    return tf.keras.Sequential([
        layers.RandomRotation(AUGMENTATION_CONFIG['ROTATION_RANGE']),
        layers.RandomZoom(AUGMENTATION_CONFIG['ZOOM_RANGE']),
        layers.RandomBrightness(AUGMENTATION_CONFIG['BRIGHTNESS_RANGE']),
        layers.RandomContrast(AUGMENTATION_CONFIG['CONTRAST_RANGE']),
    ])
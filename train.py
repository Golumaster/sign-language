import os
import tensorflow as tf
import matplotlib.pyplot as plt
import json
from data_loader import create_dataset
from model import create_model, create_data_augmentation
from config import TRAINING_CONFIG, SYSTEM_CONFIG

# Enable GPU memory growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
            if SYSTEM_CONFIG['GPU_MEMORY_LIMIT']:
                tf.config.experimental.set_virtual_device_configuration(
                    gpu,
                    [tf.config.experimental.VirtualDeviceConfiguration(
                        memory_limit=SYSTEM_CONFIG['GPU_MEMORY_LIMIT']
                    )]
                )
    except RuntimeError as e:
        print(e)

def plot_training_history(history):
    """Plot and save training metrics"""
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()

def main():
    print("Starting training process...")
    print(f"Current working directory: {os.getcwd()}")
    
    # Enable mixed precision if configured
    if SYSTEM_CONFIG['MIXED_PRECISION']:
        tf.keras.mixed_precision.set_global_policy('mixed_float16')
    
    # Create dataset pipeline
    print("\nCreating dataset pipeline...")
    try:
        dataset, class_names = create_dataset('dataset')
        
        # Split into train and validation
        dataset_size = sum(1 for _ in dataset)
        train_size = int(0.8 * dataset_size)
        
        train_dataset = dataset.take(train_size)
        val_dataset = dataset.skip(train_size)
        
        # Optimize dataset performance
        train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)
        val_dataset = val_dataset.prefetch(tf.data.AUTOTUNE)
        
    except Exception as e:
        print(f"\nError creating dataset: {str(e)}")
        print("\nPlease ensure your dataset follows this structure:")
        print("dataset/")
        print("  ├── class1/")
        print("  │   ├── image1.jpg")
        print("  │   └── image2.jpg")
        print("  └── class2/")
        print("      ├── image1.jpg")
        print("      └── image2.jpg")
        return
    
    # Save class names for prediction
    with open('class_names.json', 'w') as f:
        json.dump(class_names, f)
    
    # Create data augmentation pipeline
    data_augmentation = create_data_augmentation()
    
    # Create and compile model
    print("\nCreating model...")
    model = create_model(len(class_names))
    
    # Fixed learning rate schedule
    initial_learning_rate = TRAINING_CONFIG['INITIAL_LEARNING_RATE']
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=initial_learning_rate)
    
    if SYSTEM_CONFIG['MIXED_PRECISION']:
        optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)
    
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=TRAINING_CONFIG['EARLY_STOPPING_PATIENCE'],
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=TRAINING_CONFIG['REDUCE_LR_FACTOR'],
            patience=TRAINING_CONFIG['REDUCE_LR_PATIENCE'],
            min_lr=TRAINING_CONFIG['MIN_LEARNING_RATE'],
            verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            'best_model.h5',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]
    
    # Train model
    print("\nTraining model...")
    try:
        history = model.fit(
            train_dataset.map(
                lambda x, y: (data_augmentation(x, training=True), y),
                num_parallel_calls=tf.data.AUTOTUNE
            ),
            epochs=TRAINING_CONFIG['EPOCHS'],
            validation_data=val_dataset,
            callbacks=callbacks,
            verbose=1
        )
        
        # Plot and save training history
        plot_training_history(history)
        
        # Evaluate model
        print("\nEvaluating model...")
        test_results = model.evaluate(val_dataset)
        print(f"Test accuracy: {test_results[1]:.4f}")
        
        # Save final model
        model.save('isl_model.h5')
        print("Model saved successfully!")
        
    except Exception as e:
        print(f"\nError during training: {str(e)}")
        raise

if __name__ == "__main__":
    main()
import os
import tensorflow as tf
from config import DATASET_CONFIG

def create_dataset(data_dir, batch_size=DATASET_CONFIG['BATCH_SIZE']):
    """Create a tf.data.Dataset pipeline for efficient memory usage"""
    
    # List all class directories
    class_names = sorted(os.listdir(data_dir))
    print(f"Found {len(class_names)} classes: {class_names}")
    
    # Create class to index mapping
    class_to_idx = {cls: idx for idx, cls in enumerate(class_names)}
    
    # Collect all image paths and labels
    image_paths = []
    labels = []
    
    for class_name in class_names:
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
            
        for img_name in os.listdir(class_dir):
            if img_name.lower().endswith(('.jpg', '.jpeg')):
                image_paths.append(os.path.join(class_dir, img_name))
                labels.append(class_to_idx[class_name])
                
    print(f"Found {len(image_paths)} images total")
    
    def load_image(path, label):
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, DATASET_CONFIG['IMAGE_SIZE'])
        img = tf.cast(img, tf.float32) / 255.0
        return img, label
    
    # Create dataset
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
    dataset = dataset.shuffle(buffer_size=len(image_paths))
    dataset = dataset.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset, class_names
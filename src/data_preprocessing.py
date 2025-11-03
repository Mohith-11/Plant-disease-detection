"""
Data preprocessing utilities for Plant Disease Detection
Dataset: New Plant Diseases Dataset from Kaggle
Website: https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset
"""

import os
import shutil
import random
from pathlib import Path
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

class DataPreprocessor:
    def __init__(self, dataset_path):
        """
        Initialize data preprocessor
        
        Args:
            dataset_path (str): Path to the dataset directory
        """
        self.dataset_path = Path(dataset_path)
        self.classes = []
        
    def analyze_dataset(self):
        """Analyze the dataset structure and class distribution"""
        train_dir = self.dataset_path / "train"
        
        if not train_dir.exists():
            print(f"Training directory not found at: {train_dir}")
            return
        
        # Get all class directories
        self.classes = [d.name for d in train_dir.iterdir() if d.is_dir()]
        self.classes.sort()
        
        print(f"Found {len(self.classes)} classes:")
        print("=" * 50)
        
        total_images = 0
        class_counts = {}
        
        for class_name in self.classes:
            class_dir = train_dir / class_name
            image_count = len([f for f in class_dir.glob("*.jpg") if f.is_file()])
            class_counts[class_name] = image_count
            total_images += image_count
            print(f"{class_name}: {image_count} images")
        
        print("=" * 50)
        print(f"Total images: {total_images}")
        
        return class_counts
    
    def visualize_sample_images(self, samples_per_class=3):
        """Visualize sample images from each class"""
        train_dir = self.dataset_path / "train"
        
        if not self.classes:
            self.analyze_dataset()
        
        # Select a subset of classes for visualization
        selected_classes = self.classes[:6]  # Show first 6 classes
        
        fig, axes = plt.subplots(len(selected_classes), samples_per_class, 
                                figsize=(15, 2*len(selected_classes)))
        
        for i, class_name in enumerate(selected_classes):
            class_dir = train_dir / class_name
            image_files = list(class_dir.glob("*.jpg"))[:samples_per_class]
            
            for j, img_file in enumerate(image_files):
                try:
                    img = Image.open(img_file)
                    axes[i, j].imshow(img)
                    axes[i, j].axis('off')
                    if j == 0:
                        axes[i, j].set_ylabel(class_name.replace('___', '\n'), 
                                            rotation=0, ha='right', va='center')
                except Exception as e:
                    print(f"Error loading {img_file}: {e}")
        
        plt.tight_layout()
        plt.suptitle("Sample Images from Dataset", y=1.02, fontsize=16)
        plt.show()
    
    def create_test_split(self, test_ratio=0.1, random_seed=42):
        """
        Create a test split from the training data
        
        Args:
            test_ratio (float): Ratio of data to use for testing
            random_seed (int): Random seed for reproducibility
        """
        random.seed(random_seed)
        
        train_dir = self.dataset_path / "train"
        test_dir = self.dataset_path / "test"
        
        # Create test directory if it doesn't exist
        test_dir.mkdir(exist_ok=True)
        
        moved_count = 0
        
        for class_name in self.classes:
            train_class_dir = train_dir / class_name
            test_class_dir = test_dir / class_name
            test_class_dir.mkdir(exist_ok=True)
            
            # Get all images in the class
            image_files = list(train_class_dir.glob("*.jpg"))
            
            # Calculate number of test images
            num_test = int(len(image_files) * test_ratio)
            
            # Randomly select test images
            test_images = random.sample(image_files, num_test)
            
            # Move selected images to test directory
            for img_file in test_images:
                dest_file = test_class_dir / img_file.name
                shutil.move(str(img_file), str(dest_file))
                moved_count += 1
        
        print(f"Created test split: {moved_count} images moved to test directory")
    
    def get_class_weights(self):
        """Calculate class weights for imbalanced dataset"""
        train_dir = self.dataset_path / "train"
        class_counts = {}
        
        for class_name in self.classes:
            class_dir = train_dir / class_name
            count = len(list(class_dir.glob("*.jpg")))
            class_counts[class_name] = count
        
        # Calculate weights inversely proportional to class frequency
        total_samples = sum(class_counts.values())
        num_classes = len(class_counts)
        
        class_weights = {}
        for i, (class_name, count) in enumerate(class_counts.items()):
            weight = total_samples / (num_classes * count)
            class_weights[i] = weight
        
        return class_weights
    
    def create_data_generators(self, batch_size=32, img_size=(224, 224)):
        """
        Create data generators for training and validation
        
        Args:
            batch_size (int): Batch size for data loading
            img_size (tuple): Target image size
        
        Returns:
            train_gen, val_gen: Training and validation generators
        """
        from tensorflow.keras.preprocessing.image import ImageDataGenerator
        
        # Data augmentation for training
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest',
            validation_split=0.2
        )
        
        # Only rescaling for validation
        val_datagen = ImageDataGenerator(
            rescale=1./255,
            validation_split=0.2
        )
        
        train_generator = train_datagen.flow_from_directory(
            self.dataset_path / "train",
            target_size=img_size,
            batch_size=batch_size,
            class_mode='categorical',
            subset='training'
        )
        
        validation_generator = val_datagen.flow_from_directory(
            self.dataset_path / "train",
            target_size=img_size,
            batch_size=batch_size,
            class_mode='categorical',
            subset='validation'
        )
        
        return train_generator, validation_generator

def main():
    """Example usage of DataPreprocessor"""
    # Initialize preprocessor (update path to your dataset)
    preprocessor = DataPreprocessor("dataset")
    
    # Analyze dataset
    class_counts = preprocessor.analyze_dataset()
    
    # Visualize sample images
    # preprocessor.visualize_sample_images()
    
    # Create test split (uncomment if needed)
    # preprocessor.create_test_split(test_ratio=0.1)
    
    # Get class weights for imbalanced dataset
    # class_weights = preprocessor.get_class_weights()
    # print("Class weights:", class_weights)

if __name__ == "__main__":
    main()
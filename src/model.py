"""
Plant Disease Detection using Convolutional Neural Network (CNN)
Author: Mohith
Dataset: New Plant Diseases Dataset from Kaggle
Website: https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import os
from PIL import Image

class PlantDiseaseDetector:
    def __init__(self, img_height=224, img_width=224, num_classes=38):
        """
        Initialize the Plant Disease Detection model
        
        Args:
            img_height (int): Height of input images
            img_width (int): Width of input images
            num_classes (int): Number of disease classes
        """
        self.img_height = img_height
        self.img_width = img_width
        self.num_classes = num_classes
        self.model = None
        self.class_names = None
    
    def build_model(self):
        """Build CNN model architecture for plant disease detection"""
        self.model = keras.Sequential([
            # Data augmentation layers
            layers.RandomFlip("horizontal", input_shape=(self.img_height, self.img_width, 3)),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1),
            
            # Rescaling layer to normalize pixel values
            layers.Rescaling(1./255),
            
            # First Convolutional Block
            layers.Conv2D(32, 3, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(),
            layers.Dropout(0.25),
            
            # Second Convolutional Block
            layers.Conv2D(64, 3, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(),
            layers.Dropout(0.25),
            
            # Third Convolutional Block
            layers.Conv2D(128, 3, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(),
            layers.Dropout(0.25),
            
            # Fourth Convolutional Block
            layers.Conv2D(256, 3, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(),
            layers.Dropout(0.25),
            
            # Flatten and Dense layers
            layers.Flatten(),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        return self.model
    
    def compile_model(self, learning_rate=0.001):
        """Compile the model with optimizer, loss, and metrics"""
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy', 'top_5_accuracy']
        )
    
    def prepare_data(self, train_dir, batch_size=32, validation_split=0.2):
        """
        Prepare training and validation datasets
        
        Args:
            train_dir (str): Path to training data directory
            batch_size (int): Batch size for training
            validation_split (float): Fraction of data to use for validation
        
        Returns:
            train_ds, val_ds: Training and validation datasets
        """
        train_ds = tf.keras.utils.image_dataset_from_directory(
            train_dir,
            validation_split=validation_split,
            subset="training",
            seed=123,
            image_size=(self.img_height, self.img_width),
            batch_size=batch_size,
            label_mode='categorical'
        )
        
        val_ds = tf.keras.utils.image_dataset_from_directory(
            train_dir,
            validation_split=validation_split,
            subset="validation",
            seed=123,
            image_size=(self.img_height, self.img_width),
            batch_size=batch_size,
            label_mode='categorical'
        )
        
        # Get class names
        self.class_names = train_ds.class_names
        
        # Configure for performance
        AUTOTUNE = tf.data.AUTOTUNE
        train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
        val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
        
        return train_ds, val_ds
    
    def train(self, train_ds, val_ds, epochs=50):
        """
        Train the model
        
        Args:
            train_ds: Training dataset
            val_ds: Validation dataset
            epochs (int): Number of training epochs
        
        Returns:
            history: Training history
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        
        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                patience=10, 
                restore_best_weights=True,
                monitor='val_accuracy'
            ),
            keras.callbacks.ReduceLROnPlateau(
                factor=0.2, 
                patience=5,
                monitor='val_loss'
            )
        ]
        
        # Train the model
        history = self.model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )
        
        return history
    
    def predict_disease(self, image_path):
        """
        Predict disease from a single image
        
        Args:
            image_path (str): Path to the image file
        
        Returns:
            prediction (str): Predicted disease class
            confidence (float): Prediction confidence
        """
        if self.model is None or self.class_names is None:
            raise ValueError("Model not trained. Train the model first.")
        
        # Load and preprocess image
        img = tf.keras.utils.load_img(image_path, target_size=(self.img_height, self.img_width))
        img_array = tf.keras.utils.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)  # Create batch dimension
        
        # Make prediction
        predictions = self.model.predict(img_array)
        predicted_class_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class_idx])
        
        predicted_disease = self.class_names[predicted_class_idx]
        
        return predicted_disease, confidence
    
    def save_model(self, filepath):
        """Save the trained model"""
        if self.model is None:
            raise ValueError("No model to save.")
        self.model.save(filepath)
    
    def load_model(self, filepath):
        """Load a pre-trained model"""
        self.model = keras.models.load_model(filepath)
    
    def plot_training_history(self, history):
        """Plot training and validation accuracy/loss"""
        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        
        epochs_range = range(len(acc))
        
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, acc, label='Training Accuracy')
        plt.plot(epochs_range, val_acc, label='Validation Accuracy')
        plt.legend(loc='lower right')
        plt.title('Training and Validation Accuracy')
        
        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, loss, label='Training Loss')
        plt.plot(epochs_range, val_loss, label='Validation Loss')
        plt.legend(loc='upper right')
        plt.title('Training and Validation Loss')
        
        plt.tight_layout()
        plt.show()

def main():
    """Example usage of the PlantDiseaseDetector"""
    # Initialize the detector
    detector = PlantDiseaseDetector(img_height=224, img_width=224, num_classes=38)
    
    # Build and compile model
    detector.build_model()
    detector.compile_model(learning_rate=0.001)
    
    # Print model summary
    print("Model Architecture:")
    detector.model.summary()
    
    # Prepare data (uncomment when you have the dataset)
    # train_ds, val_ds = detector.prepare_data('dataset/train')
    
    # Train model (uncomment when ready to train)
    # history = detector.train(train_ds, val_ds, epochs=50)
    
    # Plot training history
    # detector.plot_training_history(history)
    
    # Save model
    # detector.save_model('models/plant_disease_model.h5')
    
    # Example prediction (uncomment when model is trained)
    # disease, confidence = detector.predict_disease('path/to/test/image.jpg')
    # print(f"Predicted Disease: {disease} (Confidence: {confidence:.2f})")

if __name__ == "__main__":
    main()
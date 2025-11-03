"""
Model evaluation utilities for Plant Disease Detection
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import tensorflow as tf

class ModelEvaluator:
    def __init__(self, model, class_names):
        """
        Initialize model evaluator
        
        Args:
            model: Trained Keras model
            class_names (list): List of class names
        """
        self.model = model
        self.class_names = class_names
        
    def evaluate_model(self, test_dataset, verbose=True):
        """
        Comprehensive model evaluation
        
        Args:
            test_dataset: Test dataset
            verbose (bool): Print detailed results
        
        Returns:
            dict: Evaluation metrics
        """
        # Get predictions
        y_pred_probs = self.model.predict(test_dataset, verbose=0)
        y_pred = np.argmax(y_pred_probs, axis=1)
        
        # Get true labels
        y_true = np.concatenate([y for x, y in test_dataset], axis=0)
        y_true = np.argmax(y_true, axis=1)
        
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average='weighted'
        )
        
        # Top-5 accuracy
        top5_acc = tf.keras.metrics.top_k_categorical_accuracy(
            tf.keras.utils.to_categorical(y_true, len(self.class_names)),
            y_pred_probs, k=5
        ).numpy().mean()
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'top5_accuracy': top5_acc,
            'y_true': y_true,
            'y_pred': y_pred,
            'y_pred_probs': y_pred_probs
        }
        
        if verbose:
            print("="*50)
            print("MODEL EVALUATION RESULTS")
            print("="*50)
            print(f"Accuracy: {accuracy:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall: {recall:.4f}")
            print(f"F1-Score: {f1:.4f}")
            print(f"Top-5 Accuracy: {top5_acc:.4f}")
            print("="*50)
        
        return metrics
    
    def plot_confusion_matrix(self, y_true, y_pred, normalize=True, figsize=(15, 12)):
        """
        Plot confusion matrix
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            normalize (bool): Normalize confusion matrix
            figsize (tuple): Figure size
        """
        cm = confusion_matrix(y_true, y_pred)
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            title = 'Normalized Confusion Matrix'
            fmt = '.2f'
        else:
            title = 'Confusion Matrix'
            fmt = 'd'
        
        plt.figure(figsize=figsize)
        sns.heatmap(cm, 
                   annot=True, 
                   fmt=fmt, 
                   cmap='Blues',
                   xticklabels=[name.replace('___', '\n') for name in self.class_names],
                   yticklabels=[name.replace('___', '\n') for name in self.class_names])
        
        plt.title(title, fontsize=16)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.show()
    
    def plot_classification_report(self, y_true, y_pred):
        """Plot classification report as heatmap"""
        # Generate classification report
        report = classification_report(y_true, y_pred, 
                                     target_names=self.class_names, 
                                     output_dict=True)
        
        # Convert to DataFrame for better visualization
        import pandas as pd
        df = pd.DataFrame(report).iloc[:-1, :].T  # Exclude accuracy row
        
        plt.figure(figsize=(8, 10))
        sns.heatmap(df.iloc[:-3, :3], annot=True, cmap='RdYlBu_r', fmt='.3f')
        plt.title('Classification Report', fontsize=16)
        plt.xlabel('Metrics', fontsize=12)
        plt.ylabel('Classes', fontsize=12)
        plt.tight_layout()
        plt.show()
        
        # Print detailed report
        print("\nDetailed Classification Report:")
        print("="*60)
        print(classification_report(y_true, y_pred, target_names=self.class_names))
    
    def plot_top_misclassified(self, test_dataset, y_true, y_pred, y_pred_probs, top_n=10):
        """
        Plot top misclassified examples
        
        Args:
            test_dataset: Test dataset
            y_true: True labels
            y_pred: Predicted labels
            y_pred_probs: Prediction probabilities
            top_n (int): Number of examples to show
        """
        # Find misclassified examples
        misclassified_idx = np.where(y_true != y_pred)[0]
        
        if len(misclassified_idx) == 0:
            print("No misclassified examples found!")
            return
        
        # Get confidence scores for misclassified examples
        misclassified_confidences = []
        for idx in misclassified_idx:
            confidence = y_pred_probs[idx][y_pred[idx]]
            misclassified_confidences.append((idx, confidence))
        
        # Sort by confidence (highest confidence misclassifications)
        misclassified_confidences.sort(key=lambda x: x[1], reverse=True)
        
        # Get images from dataset
        images = []
        for x, y in test_dataset:
            images.extend(x.numpy())
        
        # Plot top misclassified examples
        n_cols = 5
        n_rows = (top_n + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 3*n_rows))
        axes = axes.flatten() if n_rows > 1 else [axes]
        
        for i in range(min(top_n, len(misclassified_confidences))):
            idx, confidence = misclassified_confidences[i]
            
            # Denormalize image for display
            img = images[idx]
            if img.max() <= 1.0:
                img = (img * 255).astype(np.uint8)
            
            axes[i].imshow(img)
            axes[i].axis('off')
            
            true_class = self.class_names[y_true[idx]].replace('___', ' ')
            pred_class = self.class_names[y_pred[idx]].replace('___', ' ')
            
            axes[i].set_title(f'True: {true_class}\nPred: {pred_class}\nConf: {confidence:.3f}', 
                            fontsize=8)
        
        # Hide empty subplots
        for i in range(top_n, len(axes)):
            axes[i].axis('off')
        
        plt.suptitle('Top Misclassified Examples (High Confidence)', fontsize=16)
        plt.tight_layout()
        plt.show()
    
    def plot_per_class_accuracy(self, y_true, y_pred):
        """Plot per-class accuracy"""
        # Calculate per-class accuracy
        class_accuracies = []
        for i, class_name in enumerate(self.class_names):
            class_mask = (y_true == i)
            if np.sum(class_mask) > 0:
                class_acc = np.sum((y_true == y_pred) & class_mask) / np.sum(class_mask)
                class_accuracies.append(class_acc)
            else:
                class_accuracies.append(0.0)
        
        # Sort by accuracy
        sorted_indices = np.argsort(class_accuracies)
        sorted_classes = [self.class_names[i].replace('___', ' ') for i in sorted_indices]
        sorted_accuracies = [class_accuracies[i] for i in sorted_indices]
        
        plt.figure(figsize=(12, 8))
        bars = plt.barh(range(len(sorted_classes)), sorted_accuracies, 
                       color=['red' if acc < 0.8 else 'orange' if acc < 0.9 else 'green' 
                             for acc in sorted_accuracies])
        
        plt.yticks(range(len(sorted_classes)), sorted_classes)
        plt.xlabel('Accuracy')
        plt.title('Per-Class Accuracy')
        plt.grid(axis='x', alpha=0.3)
        
        # Add accuracy values on bars
        for i, (bar, acc) in enumerate(zip(bars, sorted_accuracies)):
            plt.text(acc + 0.01, i, f'{acc:.3f}', va='center')
        
        plt.tight_layout()
        plt.show()
    
    def generate_evaluation_report(self, test_dataset):
        """Generate comprehensive evaluation report"""
        print("Generating comprehensive evaluation report...")
        
        # Evaluate model
        metrics = self.evaluate_model(test_dataset)
        
        # Plot confusion matrix
        self.plot_confusion_matrix(metrics['y_true'], metrics['y_pred'])
        
        # Plot classification report
        self.plot_classification_report(metrics['y_true'], metrics['y_pred'])
        
        # Plot per-class accuracy
        self.plot_per_class_accuracy(metrics['y_true'], metrics['y_pred'])
        
        # Plot top misclassified examples
        self.plot_top_misclassified(test_dataset, 
                                  metrics['y_true'], 
                                  metrics['y_pred'], 
                                  metrics['y_pred_probs'])
        
        return metrics

def main():
    """Example usage of ModelEvaluator"""
    # This would be used after training a model
    # evaluator = ModelEvaluator(trained_model, class_names)
    # metrics = evaluator.generate_evaluation_report(test_dataset)
    pass

if __name__ == "__main__":
    main()
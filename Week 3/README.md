# 🌿 Week 3 – Final Model Completion (100%)

## ✔️ Overview
This week completes the entire model development pipeline for the Plant Disease Detection system.  
The work includes:

- Transfer Learning using MobileNetV2  
- Model Training and Fine-Tuning  
- Confusion Matrix and Classification Report  
- Manual Image Prediction  
- Grad-CAM Heatmap + Grad-CAM Overlay  
- Saving the Final Model  

---

## 🔥 1. Transfer Learning (MobileNetV2)
- Used pretrained **MobileNetV2** as the feature extractor.
- Added custom classification layers for 7 plant disease classes.
- Achieved **98–99% validation accuracy**.

---

## 📊 2. Model Evaluation
### Confusion Matrix  
Generated `confusion_matrix.png` to measure per-class performance.

### Classification Report  
Model produces high precision, recall, and F1 scores across all classes.

---

## 🌱 3. Manual Test Predictions
Tested the model on external leaf images (not from the dataset).  
Saved output in `sample_predictions.png`.  
Predictions were accurate and consistent.

---

## 🔥 4. Grad-CAM Explainability
Generated:
- `gradcam_heatmap.png`
- `gradcam_overlay.png`

Grad-CAM highlights the regions of the leaf the model used while predicting the disease.

---

## 📁 Files Included
This folder includes the final model artifacts, evaluation plots, sample predictions, and explainability visuals.  

# 🌱 Plant Disease Detection using CNN

## 📘 Overview

Plant diseases significantly affect crop yield and food production worldwide. Early detection of these diseases helps farmers take preventive measures to minimize losses. This project uses Convolutional Neural Networks (CNNs) to automatically classify healthy and diseased leaves from images.

The model is trained using the [New Plant Diseases Dataset](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset) from Kaggle, which contains 38 different classes of plant leaves.

## 🧠 Objective

To develop a deep learning-based image classification model capable of accurately detecting plant diseases from leaf images and improving agricultural sustainability.

### ⚙️ Features

- **Automatic plant disease classification** using CNN
- **Trained on 80,000+ labeled images** from diverse sources
- **Supports multiple plant species and diseases** (38 classes)
- **Visualization of model accuracy and loss** during training
- **Real-time prediction** from uploaded images
- **Data Augmentation**: Robust training with image augmentation techniques
- **Model Evaluation**: Comprehensive evaluation with confusion matrix, classification reports
- **Easy Prediction Interface**: Simple interface for predicting diseases from new images
- **Visualization Tools**: Built-in tools for data analysis and result visualization

## 🧩 Dataset

- **Source**: [Kaggle – New Plant Diseases Dataset](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset)
- **Classes**: 38 (healthy and diseased)
- **Total Images**: ~87,000
- **Structure**:
```
dataset/
├── train/
│   ├── Apple___Black_rot/
│   ├── Apple___healthy/
│   ├── Apple___Apple_scab/
│   ├── Apple___Cedar_apple_rust/
│   ├── Blueberry___healthy/
│   ├── Cherry_(including_sour)___healthy/
│   ├── Cherry_(including_sour)___Powdery_mildew/
│   ├── Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot/
│   ├── Corn_(maize)___Common_rust_/
│   ├── Corn_(maize)___Northern_Leaf_Blight/
│   ├── Corn_(maize)___healthy/
│   ├── Grape___Black_rot/
│   ├── Grape___Esca_(Black_Measles)/
│   ├── Grape___Leaf_blight_(Isariopsis_Leaf_Spot)/
│   ├── Grape___healthy/
│   ├── Orange___Haunglongbing_(Citrus_greening)/
│   ├── Peach___Bacterial_spot/
│   ├── Peach___healthy/
│   ├── Pepper,_bell___Bacterial_spot/
│   ├── Pepper,_bell___healthy/
│   ├── Potato___Early_blight/
│   ├── Potato___Late_blight/
│   ├── Potato___healthy/
│   ├── Raspberry___healthy/
│   ├── Soybean___healthy/
│   ├── Squash___Powdery_mildew/
│   ├── Strawberry___Leaf_scorch/
│   ├── Strawberry___healthy/
│   ├── Tomato___Bacterial_spot/
│   ├── Tomato___Early_blight/
│   ├── Tomato___Late_blight/
│   ├── Tomato___Leaf_Mold/
│   ├── Tomato___Septoria_leaf_spot/
│   ├── Tomato___Spider_mites Two-spotted_spider_mite/
│   ├── Tomato___Target_Spot/
│   ├── Tomato___Tomato_Yellow_Leaf_Curl_Virus/
│   ├── Tomato___Tomato_mosaic_virus/
│   └── Tomato___healthy/
└── valid/
    ├── Apple___Black_rot/
    ├── Apple___healthy/
    └── ...
```

## 🧰 Tech Stack

- **Language**: Python 3.8+
- **Deep Learning**: TensorFlow, Keras
- **Data Processing**: NumPy, Pandas
- **Visualization**: Matplotlib, Seaborn, Plotly
- **Image Processing**: OpenCV, PIL
- **Development**: Jupyter Notebook, VS Code
- **Dataset Source**: Kaggle

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- GPU support recommended (CUDA-compatible GPU)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Mohith-11/Plant-disease-detection.git
   cd Plant-disease-detection
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download the dataset**
   
   Download the New Plant Diseases Dataset from Kaggle:
   - Visit: [New Plant Diseases Dataset](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset)
   - Download the dataset ZIP file
   - Extract it to the `dataset/` directory in your project folder
   
   Your directory structure should look like:
   ```
   plant-disease-detection/
   ├── dataset/
   │   └── train/
   │       ├── Apple___Apple_scab/
   │       ├── Apple___Black_rot/
   │       ├── Apple___Cedar_apple_rust/
   │       ├── Apple___healthy/
   │       └── ... (other disease classes)
   ├── src/
   ├── models/
   ├── notebooks/
   └── docs/
   ```

### 🏃‍♂️ Usage

#### Training the Model

```python
from src.model import PlantDiseaseDetector

# Initialize the detector
detector = PlantDiseaseDetector(img_height=224, img_width=224, num_classes=38)

# Build and compile model
detector.build_model()
detector.compile_model(learning_rate=0.001)

# Prepare data
train_ds, val_ds = detector.prepare_data('dataset/train')

# Train the model
history = detector.train(train_ds, val_ds, epochs=50)

# Save the trained model
detector.save_model('models/plant_disease_model.h5')
```

#### Making Predictions

```python
from src.model import PlantDiseaseDetector

# Load trained model
detector = PlantDiseaseDetector()
detector.load_model('models/plant_disease_model.h5')

# Predict disease from an image
disease, confidence = detector.predict_disease('path/to/your/image.jpg')
print(f"Predicted Disease: {disease}")
print(f"Confidence: {confidence:.2f}")
```

#### Data Analysis

```python
from src.data_preprocessing import DataPreprocessor

# Analyze dataset
preprocessor = DataPreprocessor("dataset")
class_counts = preprocessor.analyze_dataset()

# Visualize sample images
preprocessor.visualize_sample_images()
```

#### Model Evaluation

```python
from src.evaluation import ModelEvaluator

# Evaluate model performance
evaluator = ModelEvaluator(model, class_names)
metrics = evaluator.generate_evaluation_report(test_dataset)
```

## 📊 Model Architecture

The CNN model features:

- **Input Layer**: 224x224x3 (RGB images)
- **Data Augmentation**: Random flip, rotation, and zoom
- **Convolutional Blocks**: 4 blocks with increasing filters (32, 64, 128, 256)
- **Regularization**: Batch normalization and dropout layers
- **Dense Layers**: Fully connected layers with dropout
- **Output**: 38 classes with softmax activation

### Model Performance

- **Accuracy**: ~95% on validation set
- **Top-5 Accuracy**: ~99%
- **Training Time**: ~2-3 hours on GPU

## 📁 Project Structure

```
plant-disease-detection/
├── 📂 src/                          # Source code
│   ├── model.py                     # Main CNN model implementation
│   ├── data_preprocessing.py        # Data loading and preprocessing
│   └── evaluation.py                # Model evaluation utilities
├── 📂 models/                       # Saved model files
├── 📂 dataset/                      # Dataset directory (create this)
│   └── train/                       # Training images (extract here)
├── 📂 notebooks/                    # Jupyter notebooks
│   └── plant_disease_detection.ipynb
├── 📂 docs/                         # Documentation
├── requirements.txt                 # Python dependencies
├── .gitignore                       # Git ignore file
└── README.md                        # This file
```

## 🛠️ Development Setup

### For Contributors

1. **Fork the repository**
2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. **Install development dependencies**
   ```bash
   pip install -r requirements.txt
   ```
4. **Run tests**
   ```bash
   python -m pytest tests/
   ```

### Adding New Features

- Create feature branches: `git checkout -b feature/your-feature-name`
- Follow PEP 8 coding standards
- Add tests for new functionality
- Update documentation as needed

## 📈 Results and Visualization

The project includes comprehensive visualization tools:

- **Training History**: Loss and accuracy curves
- **Confusion Matrix**: Per-class performance analysis
- **Classification Report**: Precision, recall, F1-score metrics
- **Misclassification Analysis**: Examples of incorrect predictions
- **Per-Class Accuracy**: Individual class performance

## 🔧 Configuration

### Model Hyperparameters

You can adjust the following parameters in `src/model.py`:

```python
# Image dimensions
IMG_HEIGHT = 224
IMG_WIDTH = 224

# Training parameters
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001

# Model architecture
NUM_CLASSES = 38
```

### Data Augmentation

Customize data augmentation in `src/data_preprocessing.py`:

```python
# Augmentation parameters
rotation_range = 20
width_shift_range = 0.2
height_shift_range = 0.2
shear_range = 0.2
zoom_range = 0.2
horizontal_flip = True
```

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Dataset**: [New Plant Diseases Dataset](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset) on Kaggle
- **Framework**: TensorFlow/Keras for deep learning implementation
- **Inspiration**: Research in agricultural AI and computer vision

## 📞 Contact

**Mohith** - [GitHub Profile](https://github.com/Mohith-11)

Project Link: [https://github.com/Mohith-11/Plant-disease-detection](https://github.com/Mohith-11/Plant-disease-detection)

## 🔮 Future Enhancements

- [ ] Deploy model as web application using Flask/Streamlit
- [ ] Mobile app development for real-time disease detection
- [ ] Integration with IoT devices for automated monitoring
- [ ] Support for additional plant species and diseases
- [ ] Real-time video processing capabilities
- [ ] Disease severity assessment and treatment recommendations

---

⭐ **Star this repository if you found it helpful!** ⭐

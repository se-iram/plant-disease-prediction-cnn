# Plant Disease Prediction Using CNN and Deep Learning

> **Semester Project - Machine Learning**  
> **Developed by:** Iram Hameed (221994), Eman Fatima (222007)

This project presents a complete **Plant Disease Prediction System** built using **Convolutional Neural Networks (CNN)** and **Deep Learning**. The system is designed to automatically identify plant diseases from leaf images and provide real-time predictions through a user-friendly **Streamlit web application**.

This is a **full-scale semester project** that demonstrates an end-to-end machine learning pipeline including dataset handling, model training, evaluation, and deployment.

---

## Project Motivation

Plant diseases are a major threat to agricultural productivity worldwide. Traditional methods of disease detection are manual, slow, and depend heavily on expert knowledge. This project aims to provide an **automated, accurate, and accessible AI-based solution** that can assist farmers, researchers, and agricultural institutions in identifying plant diseases quickly and efficiently.

---

## Project Objectives

- Develop a CNN-based deep learning model for plant disease classification.
- Train and validate the model on a large, diverse dataset of plant leaf images.
- Evaluate the model using industry-standard performance metrics.
- Deploy the trained model using a Streamlit web interface for real-time disease prediction.
- Contribute to precision agriculture through AI-driven automation.

---

## Dataset

The project uses the **PlantVillage Dataset** available on Kaggle.
Kaggle Dataset Link: https://www.kaggle.com/datasets/emmarex/plantdisease
**Key Dataset Details:**
- Total Images: ~54,000
- Number of Classes: 38
- Image Type: RGB (JPEG / PNG)
- Categories: Healthy and diseased plant leaves


**Data Split:**
- Training set: 80%
- Validation set: 20%

**Preprocessing Includes:**
- Image resizing to 224 × 224 pixels
- Normalization (pixel values scaled to [0–1])
- Data augmentation (rotation, zoom, flips, shifts) OpenCV
---

## Model Architecture

The deep learning model is built using a **Convolutional Neural Network (CNN)** architecture comprised of:

- Input Layer (224 × 224 × 3 RGB images)
- Convolutional layers (3×3 filters with ReLU)
- MaxPooling layers
- Dropout layers to prevent overfitting
- Flatten layer
- Fully connected dense layers
- Softmax output layer for multi-class classification (38 classes)

### System Architecture Diagram

https://github.com/se-iram/plant-disease-prediction-cnn/issues/1#issue-3713676287

---

## Methodology Overview

This project follows a structured and theoretical deep learning pipeline:

1. Dataset preparation and quality normalization
2. Feature extraction using CNN
3. Iterative training through supervised learning
4. Performance monitoring and validation
5. Model generalization and robustness testing
6. Integration of the trained model into a real-time web system

### Training Flow Diagram Placeholder

> **[Insert Model Training Flowchart Here]**  
> *(Data Augmentation → Training → Validation → Evaluation)*

---

## 🛠 Tools & Technologies

- **Programming Language:** Python
- **Framework:** TensorFlow / Keras
- **Libraries:** NumPy, Pandas, Matplotlib, OpenCV
- **Web Framework:** Streamlit
- **Dataset Source:** Kaggle – PlantVillage

---

## Expected Outcomes

- A highly accurate CNN model for plant disease classification
- Real-time disease prediction using web interface
- Improved accessibility of AI tools for agriculture
- A complete research-level semester project implementation

---

## Evaluation Metrics
The model is evaluated using:
- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix Visualization
  
---

### Project Type: Semester Final Project

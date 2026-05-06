# Deepfake Detection System

A deep learning-based deepfake image detection system designed to classify facial images as **Real** or **Fake** using a combination of **spatial feature learning** and **frequency-domain analysis**. The project uses a **ConvNeXt** backbone with **SRM (Spatial Rich Model) residual filters** to capture both visible facial patterns and hidden manipulation artifacts. Face cropping is used during preprocessing to reduce background noise and improve focus on manipulated facial regions.

## Project Overview

Deepfake generation techniques have become highly realistic, making visual detection increasingly difficult. This project addresses that challenge by combining:
- **Spatial-domain learning** for fine-grained facial feature extraction
- **Frequency-domain residual analysis** using SRM filters
- **Face cropping** to improve feature relevance
- **Cross-dataset evaluation** to test generalization on unseen manipulation methods

## Key Features

- **ConvNeXt-based feature extraction**
- **SRM residual frequency channel**
- **Face cropping preprocessing**
- **Multi-channel deepfake classification**
- **Cross-dataset testing for robustness**
- **Evaluation using accuracy, precision, recall, F1-score, confusion matrix, and AUC** 

## Methodology

The project was developed through systematic experimentation with multiple architectures and preprocessing strategies:
- Baseline experiments with **EfficientNet** and **Xception**
- Improved architecture using **ConvNeXt + SRM**
- Higher-resolution input experiments at **256 × 256**
- Cross-dataset validation using **Celeb-DF v2**

## Dataset

The model was trained using a merged dataset composed of:
- **FaceForensics++ (FF++) extracted facial images**
- **GAN-based deepfake dataset from Kaggle**

For cross-dataset evaluation, **Celeb-DF v2** was used to assess generalization. The dataset was split using a **70:15:15** train/validation/test ratio.

## Results

### Final Project Performance
- **Training Accuracy:** 99.79%
- **Validation Accuracy:** 75.08%
- **Test Accuracy:** 82.00%
- **Test AUC:** 0.9040

### Final Classification Report
- **Real:** Precision 0.94, Recall 0.70, F1-score 0.80
- **Fake:** Precision 0.76, Recall 0.96, F1-score 0.85
- **Overall Accuracy:** 83%

### Best Model Variation
The **ConvNeXt + SRM** setup with **256 × 256** input resolution achieved the strongest reported result in the experiments, reaching **84.99% test accuracy** and **0.9255 AUC**.

## Tech Stack

- **Python**
- **TensorFlow / Keras**
- **OpenCV**
- **NumPy**
- **Pandas**
- **Matplotlib**
- **Seaborn**
- **Scikit-learn** 

## Hardware Used

- **GPU:** NVIDIA RTX 3050 Ti
- **CPU:** AMD Ryzen 9 5900HX
- **RAM:** 16 GB
- **Storage:** SSD-based environment 

## Project Goal

The goal of this project is to build a reliable and scalable deepfake detection pipeline that improves authenticity verification in digital media by combining spatial and frequency-domain learning. 

## Conclusion

This project shows that combining a strong spatial backbone like **ConvNeXt** with **SRM-based frequency features** improves deepfake detection performance compared to earlier baseline models. The final system provides a practical approach for real vs. fake facial image classification and demonstrates improved robustness across dataset variations. 

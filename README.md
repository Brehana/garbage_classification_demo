# Garbage Classification CNN

A tiny, NPU-friendly CNN for classifying garbage into 6 categories: **cardboard, glass, metal, paper, plastic, and trash**. Trained in TensorFlow and exported as a fully-integer **int8 TFLite** model for deployment on edge devices (e.g., OpenMV with Ethos-U55 or Neural-ART NPU).

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Brehana/garbage_classification_demo/blob/main/GarbageClassification_CNN.ipynb)

---

## 📁 Repository Structure

```
├── GarbageClassification_CNN.ipynb   # Main training notebook
├── webcam_demo_Version1.py           # Webcam demo script
└── models/
    ├── custom_objects_int8.tflite    # Exported int8 TFLite model
    └── custom_objects_labels.txt     # Class label file
```

## 🚀 Getting Started

### Run in Google Colab (recommended)
Click the **Open in Colab** badge above to launch the notebook directly in your browser — no local setup required.

### Run Locally
1. Clone the repository:
   ```bash
   git clone https://github.com/Brehana/garbage_classification_demo.git
   cd garbage_classification_demo
   ```
2. Install dependencies:
   ```bash
   pip install tensorflow kagglehub matplotlib
   ```
3. Open and run `GarbageClassification_CNN.ipynb` in Jupyter.

---

## 🧠 Model Architecture

A lightweight **Depthwise Separable CNN** (`tiny_ds_cnn`) optimized for edge/NPU deployment:

- Input: `96×96` RGB image
- 3× Depthwise Separable Conv blocks with BatchNormalization
- GlobalAveragePooling → Dropout → Dense (6 classes)
- Exported as **int8 TFLite** (~11 KB)

## 📦 Dataset

Uses the [Garbage Classification dataset](https://www.kaggle.com/datasets/asdasdasasdas/garbage-classification) from Kaggle (~2,500 images across 6 classes), downloaded automatically via `kagglehub`.

## 🔧 Edge Deployment

The exported `.tflite` model is compatible with:
- **OpenMV + Ethos-U55** (via `CustomObjects_OpenMV_AE3_EthosU55.ipynb`)
- **OpenMV + Neural-ART** (via `CustomObjects_OpenMV_N6_NeuralART.ipynb`)

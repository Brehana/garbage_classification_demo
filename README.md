# Garbage Classification CNN

A tiny, NPU-friendly CNN for classifying garbage into 6 categories: **cardboard, glass, metal, paper, plastic, and trash**. Trained in TensorFlow and exported as a fully-integer **int8 TFLite** model for edge deployment on OpenMV and similar NPU-accelerated hardware.

| Notebook | Description | Launch |
|---|---|---|
| `GarbageClassification_CNN.ipynb` | Baseline tiny grayscale CNN (~2k params) for edge/NPU deployment | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Brehana/garbage_classification_demo/blob/main/GarbageClassification_CNN.ipynb) |
| `GarbageClassification_CNN_Improved.ipynb` | Improved RGB CNN with BatchNorm, augmentation & LR callbacks | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Brehana/garbage_classification_demo/blob/main/GarbageClassification_CNN_Improved.ipynb) |

---

## 📁 Repository Structure

```
├── GarbageClassification_CNN.ipynb           # Baseline training notebook (tiny, edge-friendly)
├── GarbageClassification_CNN_Improved.ipynb  # Improved teaching notebook (RGB, BatchNorm, callbacks)
├── webcam_demo_Version1.py                   # Webcam demo script
└── models/
    ├── custom_objects_int8.tflite            # Exported int8 TFLite model (baseline)
    ├── teaching_cnn_int8.tflite              # Exported int8 TFLite model (improved)
    ├── custom_objects_labels.txt             # Class label file (baseline)
    └── teaching_cnn_labels.txt              # Class label file (improved)
```

## 🚀 Getting Started

### Run in Google Colab (recommended)
Click an **Open in Colab** badge in the table above to launch either notebook directly in your browser — no local setup required.

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
3. Open and run either notebook in Jupyter.

---

## 🧠 Model Architectures

### Baseline (`GarbageClassification_CNN.ipynb`)
A lightweight **Depthwise Separable CNN** (`tiny_ds_cnn`) optimized for edge/NPU deployment:
- Input: `96×96` grayscale image (1 channel)
- 3× Depthwise Separable Conv blocks
- GlobalAveragePooling → Dense (6 classes)
- Exported as **int8 TFLite** (~11 KB)

### Improved (`GarbageClassification_CNN_Improved.ipynb`)
A stronger **teaching CNN** designed for better accuracy:
- Input: `96×96` RGB image (3 channels)
- Conv → BatchNorm → ReLU blocks with increasing filters (32 → 64 → 128)
- GlobalAveragePooling → Dropout → Dense (6 classes)
- Data augmentation, ReduceLROnPlateau & EarlyStopping callbacks
- Training history plots (accuracy & loss) + confusion matrix

## 📦 Dataset

Uses the [Garbage Classification dataset](https://www.kaggle.com/datasets/asdasdasasdas/garbage-classification) from Kaggle (~2,500 images across 6 classes), downloaded automatically via `kagglehub`.

## 🔧 Edge Deployment

The exported `.tflite` model is compatible with:
- **OpenMV + Ethos-U55** (via `CustomObjects_OpenMV_AE3_EthosU55.ipynb`)
- **OpenMV + Neural-ART** (via `CustomObjects_OpenMV_N6_NeuralART.ipynb`)

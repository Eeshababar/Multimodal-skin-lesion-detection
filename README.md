# Multimodal Disease Classification with CNNs and Fusion Strategies

![Python](https://img.shields.io/badge/Python-3.9-blue)  
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)  
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5-green)  
![Status](https://img.shields.io/badge/Status-Research--Prototype-yellow)

---

## 📌 Overview
This project explores **multimodal fusion techniques** for disease classification using medical images and patient metadata.  
We evaluate two approaches:

- **Early Fusion** → Concatenate features from both modalities before classification  
- **Late Fusion** → Train separate classifiers for each modality and ensemble their predictions  

Two CNN backbones were used:
- **ResNet-50**
- **EfficientNet-B0**

### Included Experiments
- `efficientnet-and-early-fusion.ipynb` → EfficientNet-B0 + Early Fusion  
- `resnet50-and-early-fusion.ipynb` → ResNet-50 + Early Fusion  
- `resnet-50-and-late-fusion.ipynb` → ResNet-50 + Late Fusion  

---

## 🚀 Key Features
- **Image Branch** → Extracts features from medical images using pretrained CNNs (ResNet50, EfficientNet-B0).  
- **Metadata Branch** → Encodes patient demographics/clinical data with ML models (XGBoost, Logistic Regression, CatBoost, etc.).  
- **Fusion Strategies**  
  - *Early Fusion*: Feature concatenation + joint classifier  
  - *Late Fusion*: Probability ensembling (stacking/averaging)  
- **Performance**: Best result ~**79% accuracy** with ResNet50 + XGBoost using late fusion  

---

| Backbone     | Fusion Strategy | Method                          | Accuracy |
| ------------ | --------------- | ------------------------------- | -------- |
| ResNet-50    | Early Fusion    | ResNet-50 + ML classifier       | \~75%    |
| ResNet-50    | Late Fusion     | ResNet-50 + XGBoost (stacking)  | **79%**  |
| EfficientNet | Early Fusion    | EfficientNet-B0 + ML classifier | \~73%    |


## 📂 Project Structure

├── efficientnet-and-early-fusion.ipynb # EfficientNet-B0 + Early Fusion
├── resnet50-and-early-fusion.ipynb # ResNet-50 + Early Fusion
├── resnet-50-and-late-fusion.ipynb # ResNet-50 + Late Fusion
├── requirements.txt # Dependencies
├── README.md # Project documentation
└── data/ # Dataset folder (not included)

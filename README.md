# Brain Tumor MRI Classification

## Description
This project implements a deep learning pipeline for **brain tumor classification** using MRI images. It includes both a **Custom CNN** and **pretrained models** (EfficientNetB0, InceptionV3, ResNet50) with ImageNet weights. The application allows users to upload MRI scans and predicts the tumor type, visualizes Grad-CAM heatmaps, displays confidence scores, and compares predictions from multiple models.  

---

## About the Dataset
- **Dataset:** Brain MRI Images for Tumor Classification  
- **Categories:**  
  - Glioma  
  - Meningioma  
  - Pituitary Tumor  
  - No Tumor  
- **Source:** Public datasets available from medical imaging repositories.  
- **Size:** Approximately 3,000  
- **Preprocessing:**  
  - Resizing to 224x224 pixels  
  - Normalization of pixel values  
  - Optional data augmentation for training  

---

## Tools Used
- **Programming Language:** Python 3.x  
- **Deep Learning Libraries:** TensorFlow, Keras  
- **Image Processing:**  NumPy  
- **Visualization:** Matplotlib  
- **Web Deployment:** Streamlit  
- **Others:** Colab / Jupyter Notebook for experimentation  

---

## Deep learning Architectures Used:
- **Custom Convolutional Neural Network (CNN):**  
  - Multiple convolutional layers with ReLU activation  
  - Batch Normalization  
  - Dropout for regularization  
  - Fully connected dense layers for classification  
- **Transfer Learning Models:**  
  - **EfficientNetB0**  
  - **InceptionV3**  
  - **ResNet50**  
  - Pretrained on ImageNet, with top layers replaced for 4-class classification  
- **Grad-CAM:** Visual explanations of model predictions highlighting regions of interest  

---

## Potential Use Cases
- **Medical Diagnosis Assistance:** Helps radiologists quickly identify tumor types.  
- **Educational Tool:** Demonstrates deep learning on medical imaging datasets.  
- **Research:** Can be extended for tumor segmentation or severity assessment.  

---

## Data Pipeline
1. **Data Collection:** Collect MRI images categorized into tumor types.  
2. **Preprocessing:**  
   - Resize images to 224x224  
   - Normalize pixel values to [0,1]  
   - Data augmentation (optional)  
3. **Model Training:**  
   - Train Custom CNN from scratch  
   - Fine-tune pretrained models using transfer learning  
   - Save best models using `ModelCheckpoint`  
4. **Evaluation:**  
   - Metrics: Accuracy, Precision, Recall, F1-score  
   - Visualization: Confusion matrix, training history plots  
5. **Deployment:**  
   - Build Streamlit application for interactive predictions  
   - Upload MRI images and visualize Grad-CAM, confidence scores, and model comparisons  

---

## Future Improvements
- **Full Multi-Model Deployment:** Integrate all models (Custom CNN + pretrained) live for predictions.  
- **Segmentation:** Extend to brain tumor localization/segmentation instead of only classification.  
- **Explainability:** Improve Grad-CAM to show finer resolution heatmaps.  
- **Mobile/Web Deployment:** Convert to a mobile-friendly app or full web service.  
- **Data Expansion:** Include more MRI images from different hospitals for better generalization.  

---

## How to Run
1. Clone the repository:  
   ```bash
   git clone https://github.com/username/brain-tumor-mri-classification.git

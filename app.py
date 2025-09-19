import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
import base64
from tensorflow.keras.applications import (
    EfficientNetB0, InceptionV3, ResNet50,
    efficientnet, inception_v3, resnet50
)

# -------------------------------
# Initialize shared variables
# -------------------------------
img_rgb, img_resized, img_input, preds, model = None, None, None, None, None

# Set wide layout and title
st.set_page_config(page_title="Brain-Tumour MRI Image Classification", layout="wide")
st.title("Brain Tumour Detector APP ✨")

# -------------------------------
# Background image helper
# -------------------------------
def set_bg_from_local(image_path):
    with open(image_path, "rb") as file:
        encoded = base64.b64encode(file.read()).decode()
    css = f"""
    <style>
    .stApp {{
        background-image: linear-gradient(rgba(255, 255, 255, 0.6), rgba(255, 255, 255, 0.6)), url("data:image/jpg;base64,{encoded}");
        background-size: cover;
        background-repeat: no-repeat;
        background-position: center;
        color: black;
    }}
    h1, h2, h3, h4, h5, h6, .stMarkdown, .css-1v0mbdj p {{
        color: black !important;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

image_path = r"D:\\DataScience\\Guvi_projects\\BrainTumor_MRI_Image_Classification\\photo.jpg"
set_bg_from_local(image_path)

# -------------------------------
# 1️⃣ Load Models
# -------------------------------
@st.cache_resource
def load_all_models():
    model_custom = tf.keras.models.load_model("models/best_custom_cnn.h5")
    model_en = tf.keras.models.load_model("models/best_EfficientNetB0.keras")
    model_incep = tf.keras.models.load_model("models/best_InceptionV3.keras")
    model_resnet = tf.keras.models.load_model("models/ResNet50_final_hypertuned.h5")
    return model_custom, model_en, model_incep, model_resnet

model_custom, model_en, model_incep, model_resnet = load_all_models()
class_names = ['glioma', 'meningioma', 'no_tumor', 'pituitary']

# -------------------------------
# 2️⃣ Image preprocessing
# -------------------------------
def preprocess_image(uploaded_file, model_name):
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (224, 224))

    if model_name == "Custom CNN":
        img_input = np.expand_dims(img_resized, axis=0)
    elif model_name == "EfficientNetB0":
        img_input = efficientnet.preprocess_input(img_resized)
        img_input = np.expand_dims(img_input, axis=0)
    elif model_name == "InceptionV3":
        img_input = inception_v3.preprocess_input(img_resized)
        img_input = np.expand_dims(img_input, axis=0)
    elif model_name == "ResNet50":
        img_input = resnet50.preprocess_input(img_resized)
        img_input = np.expand_dims(img_input, axis=0)
    else:
        img_input = np.expand_dims(img_resized, axis=0)

    return img_rgb, img_resized, img_input


# -------------------------------
# 3️⃣ Tabs
# -------------------------------
tabs = st.tabs(["About", "Prediction", "Confidence Scores"])

# ---- Tab 0: About ----
with tabs[0]:
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        ## Project Title  
        🧠 **Brain Tumor MRI Image Classification**

        ### Skills & Takeaways
        - Data Preprocessing & Augmentation  
        - Deep Learning with TensorFlow/Keras  
        - Convolutional Neural Networks (CNNs)  
        - Streamlit App Deployment  

        ### Dataset  
        📂 **Brain MRI Dataset**  
        - MRI images with/without tumors  
        - Labeled into multiple categories  
        - Preprocessed to a fixed size & normalized  
        - Augmented to improve model generalization  

        ### Algorithms & Techniques  
        🤖 **Deep Learning Approaches**  
        - Convolutional Neural Networks (CNN)  
        - Transfer Learning (ResNet / EfficientNet / Inception)  
        """)
    with col2:
        st.markdown("""
        ### Business Use Cases
        - Early Detection & Diagnosis of Brain Tumors  
        - Assist Radiologists with MRI Analysis  
        - Reduce Human Error in Image Interpretation  
        - Support Research in Medical Imaging AI  

        ### Tools Used  
        - TensorFlow / Keras  
        - NumPy, OpenCV  
        - Matplotlib  
        - Streamlit  

        ### Future Improvements  
        🚀 **Planned Enhancements**  
        - Use larger & more diverse datasets  
        - Explore advanced architectures (Vision Transformers)  
        - Hyperparameter tuning for higher accuracy  
        - Mobile/edge deployment for real-time MRI scans  
        """)

# ---- Tab 1: Prediction ----
with tabs[1]:
    st.subheader("Prediction")
    uploaded_file = st.file_uploader("Upload a brain MRI image", type=["jpg", "png", "jpeg"])

    if uploaded_file:
        model_option = st.selectbox(
            "Select model",
            ["Custom CNN", "EfficientNetB0", "InceptionV3", "ResNet50"]
        )

        if st.button("Submit for Prediction"):
            img_rgb, img_resized, img_input = preprocess_image(uploaded_file, model_option)

            # Select model
            if model_option == "Custom CNN":
                model = model_custom
            elif model_option == "EfficientNetB0":
                model = model_en
            elif model_option == "InceptionV3":
                model = model_incep
            else:
                model = model_resnet

            preds = model.predict(img_input)
            class_idx = np.argmax(preds)
            confidence = preds[0][class_idx] * 100

            col1, col2 = st.columns([1, 2])
            with col1:
                st.image(img_rgb, caption="Uploaded Image", width=200)
            with col2:
                st.markdown(f"""
                ### 🧠 Prediction Result  
                - **Tumor Type:** `{class_names[class_idx]}`  
                - **Confidence:** `{confidence:.2f}%`
                """)

# ---- Tab 2: Confidence Scores ----
with tabs[2]:
    st.subheader("Confidence Scores (All Classes)")
    col1,col2 = st.columns(2)
    with col1:
        if preds is not None:
            for i, score in enumerate(preds[0]):
                st.write(f"**{class_names[i]}**")
                st.progress(float(score))
        else:
            st.warning("Please upload an image and get prediction first.")
    with col2:
        if preds is not None:
            fig, ax = plt.subplots()
            ax.bar(class_names, preds[0] * 100)
            ax.set_ylabel("Confidence (%)")
            ax.set_title("Model Confidence per Class")
            st.pyplot(fig)
        else:
            st.warning("Please upload an image and get prediction first.")

import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import os

# --------------------
# Streamlit Page Config
# --------------------
st.set_page_config(page_title="Tuberculosis Detection", layout="centered")
st.title("🫁 Tuberculosis Detection from Chest X-rays")
st.write("Upload a chest X-ray image and choose the model to detect Tuberculosis.")

# --------------------
# Model Loader
# --------------------
@st.cache_resource
def load_trained_model(model_path):
    if not os.path.exists(model_path):
        st.error(f"Model file not found: {model_path}")
        st.stop()
    return load_model(model_path)

# Mapping model names to file paths
model_files = {
    "ResNet50": "ResNet50_best.h5",
    "VGG16": "VGG16_best.h5",
    "EfficientNetB0": "EfficientNetB0_best.h5"
}

# Model selection
selected_model_name = st.selectbox("Select Model", list(model_files.keys()))
model_path = model_files[selected_model_name]
model = load_trained_model(model_path)

# --------------------
# Preprocessing Function
# --------------------
def preprocess_image(img):
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

# --------------------
# File Upload & Prediction
# --------------------
uploaded_file = st.file_uploader("📤 Upload a chest X-ray image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)
    st.write("Processing image...")

    processed_img = preprocess_image(img)
    prediction = model.predict(processed_img)

    # Assuming binary classification with sigmoid output
    class_names = ["Normal", "Tuberculosis"]
    prob = prediction[0][0]
    predicted_class = class_names[int(prob > 0.5)]
    confidence = prob if predicted_class == "Tuberculosis" else 1 - prob

    st.success(f"**Prediction:** {predicted_class}")
    st.info(f"**Confidence:** {confidence * 100:.2f}%")

    st.write(f"**Model Used:** {selected_model_name}")

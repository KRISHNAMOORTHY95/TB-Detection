# app.py
import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# Page config
st.set_page_config(page_title="TB Detection App", page_icon="🫁", layout="centered")

st.title("🫁 Tuberculosis Detection using ResNet50")
st.write("Upload a chest X-ray image, and the model will classify it as **TB** or **Normal**.")

# Load model (cache to avoid reloading every time)
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("ResNet50_best.h5")
    return model

model = load_model()

# Preprocess image
def preprocess_image(img: Image.Image):
    img = img.convert("RGB")
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

uploaded_file = st.file_uploader("Upload X-ray Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded X-ray", use_column_width=True)

    if st.button("Predict"):
        with st.spinner("Classifying..."):
            img_array = preprocess_image(image)
            prediction = model.predict(img_array)[0][0]
            label = "TB" if prediction >= 0.5 else "Normal"
            confidence = prediction if label == "TB" else 1 - prediction

            st.subheader(f"Prediction: **{label}**")
            st.write(f"Confidence: **{confidence * 100:.2f}%**")

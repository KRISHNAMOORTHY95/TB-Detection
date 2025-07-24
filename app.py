import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import cv2

# Load the trained VGG16 model
model = load_model(r'C:\Users\Pitchamani\Desktop\TB_Test\env\Scripts\VGG16_model.h5')

def preprocess_image(img):
    img_uint8 = (img * 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img = clahe.apply(img_uint8)
    img = cv2.GaussianBlur(img, (5, 5), 0)
    return img / 255.0

st.title("TB Detection from Chest X-ray")
uploaded_file = st.file_uploader("Upload an X-ray image", type=['jpg'])

if uploaded_file:
    img = Image.open(uploaded_file).convert('L')
    img = img.resize((224, 224))
    img_array = np.array(img)
    img_array = preprocess_image(img_array)
    img_array = img_array.reshape(1, 224, 224, 1)
    img_array_3ch = np.repeat(img_array, 3, axis=-1)
    prediction = model.predict(img_array_3ch)
    prob = prediction[0][1]
    result = "TB Positive" if prob > 0.5 else "Normal"
    st.image(img, caption="Uploaded Image", use_container_width=True)
    st.write(f"Prediction: {result} (Probability of TB: {prob:.4f})")
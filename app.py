# app/app.py
import streamlit as st
import numpy as np
from PIL import Image
import os
import pandas as pd

st.set_page_config(page_title="TB Detection", page_icon="🫁", layout="centered")
st.title("🫁 Tuberculosis Detection from Chest X-rays")

TFLITE_PATH = "tflite_models/ResNet50_best.tflite"
H5_PATH = "models/ResNet50_best.h5"

# Try to load TFLite first (lightweight)
use_tflite = False
interpreter = None
input_details = None
output_details = None

if os.path.exists(TFLITE_PATH):
    try:
        import tflite_runtime.interpreter as tflite  # preferred
    except Exception:
        try:
            import tensorflow as tf
            tflite = tf.lite
        except Exception:
            tflite = None

    if tflite is not None:
        interpreter = tflite.Interpreter(model_path=TFLITE_PATH)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        use_tflite = True

# If TFLite not available, try Keras h5
model = None
if not use_tflite:
    if os.path.exists(H5_PATH):
        try:
            import tensorflow as tf
            model = tf.keras.models.load_model(H5_PATH, compile=False)
        except Exception as e:
            st.error(f"Failed to load H5 model: {e}")
    else:
        st.warning("No model found. Place a TFLite model in tflite_models/ or an H5 model in models/")

def preprocess_image(img: Image.Image):
    img = img.convert("RGB").resize((224,224))
    arr = np.array(img, dtype=np.float32)/255.0
    arr = np.expand_dims(arr, axis=0)
    return arr

def predict_tflite(interpreter, input_details, output_details, arr):
    interpreter.set_tensor(input_details[0]['index'], arr.astype(np.float32))
    interpreter.invoke()
    out = interpreter.get_tensor(output_details[0]['index'])
    return float(out.ravel()[0])

def predict_h5(model, arr):
    p = model.predict(arr)[0][0]
    return float(p)

uploaded = st.file_uploader("Upload chest X-ray (JPG/PNG/TIFF)", type=['jpg','jpeg','png','tiff'])
if uploaded:
    img = Image.open(uploaded)
    st.image(img, use_column_width=True, caption="Uploaded X-ray")
    if st.button("Analyze"):
        arr = preprocess_image(img)
        try:
            if use_tflite and interpreter is not None:
                pred = predict_tflite(interpreter, input_details, output_details, arr)
            elif model is not None:
                pred = predict_h5(model, arr)
            else:
                st.error("No model available")
                pred = None

            if pred is not None:
                tb_prob = pred
                normal_prob = 1 - tb_prob
                is_tb = tb_prob >= 0.5
                st.subheader("Results")
                if is_tb:
                    st.error("🚨 TB Detected")
                else:
                    st.success("✅ Normal")
                col1, col2 = st.columns(2)
                col1.metric("TB Probability", f"{tb_prob*100:.1f}%")
                col2.metric("Normal Probability", f"{normal_prob*100:.1f}%")
                df = pd.DataFrame({'Condition': ['TB','Normal'], 'Probability': [tb_prob, normal_prob]}).set_index('Condition')
                st.bar_chart(df)
                with st.expander("Detailed"):
                    st.write(f"Raw prediction score: {tb_prob:.4f}")
        except Exception as e:
            st.error(f"Error during prediction: {e}")

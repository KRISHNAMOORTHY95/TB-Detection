import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import pandas as pd

# Page config
st.set_page_config(
    page_title="TB Detection App", 
    page_icon="🫁", 
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Title and description
st.title("🫁 Tuberculosis Detection using ResNet50")
st.markdown("""
Upload a chest X-ray image, and the AI model will classify it as **TB** or **Normal**.

⚠️ **Medical Disclaimer**: This tool is for educational purposes only and should not be used for actual medical diagnosis. 
Always consult healthcare professionals for medical advice.
""")

# Load model with error handling
@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model("ResNet50_best.h5")
        return model, None
    except Exception as e:
        return None, str(e)

# Initialize model
model, error = load_model()

if error:
    st.error(f"❌ Failed to load model: {error}")
    st.stop()
else:
    st.success("✅ Model loaded successfully!")

# Preprocess image with validation
def preprocess_image(img: Image.Image):
    try:
        if img.mode != 'RGB':
            img = img.convert("RGB")
        img = img.resize((224, 224), Image.Resampling.LANCZOS)
        img_array = np.array(img, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        return img_array, None
    except Exception as e:
        return None, str(e)

def validate_image(uploaded_file):
    try:
        if uploaded_file.size > 10 * 1024 * 1024:
            return False, "File too large. Please upload an image smaller than 10MB."
        image = Image.open(uploaded_file)
        width, height = image.size
        if width < 100 or height < 100:
            return False, "Image too small. Please upload a higher resolution X-ray image."
        return True, None
    except Exception as e:
        return False, f"Invalid image file: {str(e)}"

# File uploader
st.subheader("📁 Upload X-ray Image")
uploaded_file = st.file_uploader(
    "Choose an X-ray image file", 
    type=["jpg", "jpeg", "png", "bmp", "tiff"],
    help="Supported formats: JPG, JPEG, PNG, BMP, TIFF"
)

if uploaded_file is not None:
    is_valid, validation_error = validate_image(uploaded_file)
    
    if not is_valid:
        st.error(f"❌ {validation_error}")
    else:
        try:
            image = Image.open(uploaded_file)
            
            col1, col2 = st.columns([1, 1])
            with col1:
                st.image(image, caption="Original X-ray", use_column_width=True)
            with col2:
                st.info(f"""
                **Image Info:**
                - Size: {image.size[0]} x {image.size[1]} pixels
                - Mode: {image.mode}
                - Format: {image.format}
                - File size: {uploaded_file.size / 1024:.1f} KB
                """)
            
            if st.button("🔬 Analyze X-ray", type="primary", use_container_width=True):
                with st.spinner("🧠 AI is analyzing the X-ray image..."):
                    img_array, preprocess_error = preprocess_image(image)
                    
                    if preprocess_error:
                        st.error(f"❌ Error preprocessing image: {preprocess_error}")
                    else:
                        try:
                            prediction = model.predict(img_array, verbose=0)[0][0]
                            
                            tb_prob = float(prediction)
                            normal_prob = float(1 - prediction)
                            is_tb = tb_prob >= 0.5
                            label = "TB Detected" if is_tb else "Normal"
                            
                            st.divider()
                            st.subheader("🔍 Analysis Results")
                            
                            if is_tb:
                                st.error(f"🚨 **{label}**")
                            else:
                                st.success(f"✅ **{label}**")
                            
                            colA, colB = st.columns(2)
                            colA.metric("TB Probability", f"{tb_prob * 100:.1f}%")
                            colB.metric("Normal Probability", f"{normal_prob * 100:.1f}%")
                            
                            # Bar chart visualization
                            prob_df = pd.DataFrame({
                                'Condition': ['TB', 'Normal'],
                                'Probability': [tb_prob, normal_prob]
                            })
                            prob_df.set_index('Condition', inplace=True)
                            st.bar_chart(prob_df)
                            
                            with st.expander("📊 Detailed Results"):
                                st.write(f"**Raw Prediction Score:** {tb_prob:.4f}")
                                st.write(f"**Threshold:** 0.5")
                                st.write(f"**Classification:** {'TB' if is_tb else 'Normal'}")
                                
                                if is_tb:
                                    st.warning("""
                                    **⚠️ Important:**
                                    - This result suggests possible TB indicators
                                    - Please consult a medical professional immediately
                                    - Further clinical examination and tests are required
                                    - This AI tool is not a substitute for professional medical diagnosis
                                    """)
                                else:
                                    st.info("""
                                    **ℹ️ Note:**
                                    - The image appears normal according to the AI model
                                    - This does not guarantee absence of disease
                                    - Regular medical check-ups are still recommended
                                    - Consult healthcare professionals for any concerns
                                    """)
                        
                        except Exception as e:
                            st.error(f"❌ Error during prediction: {str(e)}")
        
        except Exception as e:
            st.error(f"❌ Error loading image: {str(e)}")

# Sidebar
with st.sidebar:
    st.header("ℹ️ About")
    st.write("""
    This application uses a ResNet50 deep learning model trained on chest X-ray images to detect tuberculosis.
    
    **How it works:**
    1. Upload a chest X-ray image
    2. The AI model analyzes the image
    3. Get classification results with confidence score
    """)
    
    st.header("📋 Usage Tips")
    st.write("""
    **For best results:**
    - Use clear, high-quality X-ray images
    - Ensure proper lighting and contrast
    - Upload standard chest X-ray views
    - File size should be reasonable (< 10MB)
    """)

st.divider()
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <small>
        🏥 TB Detection AI Tool | 
        ⚠️ Not for medical diagnosis | 
        🔬 Educational purposes only
    </small>
</div>
""", unsafe_allow_html=True)

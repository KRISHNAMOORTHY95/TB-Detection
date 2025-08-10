import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import io

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
        # Convert to RGB if not already
        if img.mode != 'RGB':
            img = img.convert("RGB")
        
        # Resize to model input size
        img = img.resize((224, 224), Image.Resampling.LANCZOS)
        
        # Convert to array and normalize
        img_array = np.array(img, dtype=np.float32) / 255.0
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array, None
    except Exception as e:
        return None, str(e)

def validate_image(uploaded_file):
    """Validate uploaded file is a proper image"""
    try:
        # Check file size (limit to 10MB)
        if uploaded_file.size > 10 * 1024 * 1024:
            return False, "File too large. Please upload an image smaller than 10MB."
        
        # Try to open as image
        image = Image.open(uploaded_file)
        
        # Check if it's a reasonable size for X-ray
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
    # Validate the uploaded file
    is_valid, validation_error = validate_image(uploaded_file)
    
    if not is_valid:
        st.error(f"❌ {validation_error}")
    else:
        try:
            # Display the uploaded image
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
            
            # Prediction button
            if st.button("🔬 Analyze X-ray", type="primary", use_container_width=True):
                with st.spinner("🧠 AI is analyzing the X-ray image..."):
                    # Preprocess image
                    img_array, preprocess_error = preprocess_image(image)
                    
                    if preprocess_error:
                        st.error(f"❌ Error preprocessing image: {preprocess_error}")
                    else:
                        try:
                            # Make prediction
                            prediction = model.predict(img_array, verbose=0)[0][0]
                            
                            # Determine label and confidence
                            is_tb = prediction >= 0.5
                            label = "TB Detected" if is_tb else "Normal"
                            confidence = prediction if is_tb else 1 - prediction
                            
                            # Display results
                            st.divider()
                            st.subheader("🔍 Analysis Results")
                            
                            # Create columns for results
                            result_col1, result_col2 = st.columns([1, 1])
                            
                            with result_col1:
                                if is_tb:
                                    st.error(f"🚨 **{label}**")
                                else:
                                    st.success(f"✅ **{label}**")
                            
                            with result_col2:
                                st.metric("Confidence", f"{confidence * 100:.1f}%")
                            
                            # Confidence bar
                            st.progress(confidence, text=f"Model Confidence: {confidence * 100:.1f}%")
                            
                            # Additional information
                            with st.expander("📊 Detailed Results"):
                                st.write(f"**Raw Prediction Score:** {prediction:.4f}")
                                st.write(f"**Threshold:** 0.5")
                                st.write(f"**Classification:** {'TB' if prediction >= 0.5 else 'Normal'}")
                                
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
                            st.write("Please try again or contact support if the issue persists.")
        
        except Exception as e:
            st.error(f"❌ Error loading image: {str(e)}")

# Sidebar with information
with st.sidebar:
    st.header("ℹ️ About")
    st.write("""
    This application uses a ResNet50 deep learning model trained on chest X-ray images to detect tuberculosis.
    
    **How it works:**
    1. Upload a chest X-ray image
    2. The AI model analyzes the image
    3. Get classification results with confidence score
    
    **Model Details:**
    - Architecture: ResNet50
    - Input size: 224x224 pixels
    - Training data: Chest X-ray images
    
    **Limitations:**
    - For educational use only
    - Not for medical diagnosis
    - Accuracy depends on image quality
    - Should not replace professional medical advice
    """)
    
    st.header("📋 Usage Tips")
    st.write("""
    **For best results:**
    - Use clear, high-quality X-ray images
    - Ensure proper lighting and contrast
    - Upload standard chest X-ray views
    - File size should be reasonable (< 10MB)
    
    **Supported formats:**
    - JPG, JPEG, PNG, BMP, TIFF
    """)

# Footer
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

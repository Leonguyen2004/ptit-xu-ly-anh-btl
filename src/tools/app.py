import streamlit as st
from PIL import Image
import cv2
import numpy as np
from pathlib import Path
import base64

# Import backend logic
from utils import LicensePlateRecognizer

# Page Configuration
st.set_page_config(
    page_title="License Plate Recognition",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load Custom CSS
def load_css(file_path):
    with open(file_path) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

css_path = Path(__file__).parent / "style.css"
if css_path.exists():
    load_css(css_path)

# Initialize Recognizer
@st.cache_resource
def get_recognizer():
    return LicensePlateRecognizer()

try:
    recognizer = get_recognizer()
except Exception as e:
    st.error(f"Failed to load models: {e}")
    st.stop()

# Sidebar
with st.sidebar:
    st.title("⚙️ Settings")
    
    st.markdown("### Model Parameters")
    yolo_conf = st.slider("YOLO Confidence", 0.0, 1.0, 0.25, 0.05, help="Minimum confidence for character detection")
    
    st.markdown("### View Options")
    show_debug = st.toggle("Debug Mode", False, help="Show individual character crops and confidence scores")
    
    st.markdown("---")
    st.markdown("### About")
    st.info(
        "This tool uses a YOLOv8 model for character detection and a custom CNN for character classification."
    )

# Main Content
st.title("🚗 License Plate Recognition")
st.markdown("Upload a license plate image to extract the text.")

# File Uploader
uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png', 'bmp', 'webp'])

if uploaded_file is not None:
    # Display columns
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### Original Image")
        image = Image.open(uploaded_file)
        st.image(image, use_column_width=True)

    # Process Button
    if st.button("🔍 Analyze Image", type="primary", use_container_width=True):
        with st.spinner("Processing..."):
            try:
                # Run processing
                result = recognizer.process_image(image, yolo_conf=yolo_conf)
                
                if isinstance(result, tuple): # Error case
                    st.error(result[1])
                else:
                    # Success
                    text = result['text']
                    annotated_img = result['annotated_image']
                    debug_crops = result['debug_crops']
                    
                    with col2:
                        st.markdown("### Result")
                        st.image(annotated_img, use_column_width=True)
                        
                        st.success("Recognition Complete!")
                        st.metric(label="License Plate Number", value=text)

                    # Debug View
                    if show_debug:
                        st.markdown("---")
                        st.markdown("### 🐞 Debug Info")
                        st.markdown(f"Found {len(debug_crops)} characters.")
                        
                        # Display crops in a grid
                        cols = st.columns(len(debug_crops) if len(debug_crops) > 0 else 1)
                        for i, crop_data in enumerate(debug_crops):
                            with cols[i]:
                                st.image(crop_data['image'], width=50)
                                st.caption(f"**{crop_data['char']}**\n{crop_data['conf']*100:.1f}%")

            except Exception as e:
                st.error(f"An error occurred during processing: {e}")
                import traceback
                st.code(traceback.format_exc())

else:
    # Empty state
    st.info("👆 Please upload an image to get started.")

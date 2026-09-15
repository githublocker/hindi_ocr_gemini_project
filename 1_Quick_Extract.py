import streamlit as st
from PIL import Image
import re
from surya.recognition import RecognitionPredictor
from surya.inference import SuryaInferenceManager

st.set_page_config(layout="wide")

@st.cache_resource
def load_surya_models():
    manager = SuryaInferenceManager()
    recognizer = RecognitionPredictor(manager)
    return recognizer

recognizer = load_surya_models()

st.title("Abhishek_Hindi_English_OCR")
uploaded_file = st.file_uploader("Upload Image")

if uploaded_file:
    image = Image.open(uploaded_file)
    
    predictions = recognizer([image])
    
    raw_html_text = " ".join([block.html for block in predictions[0].blocks if block.html])
    clean_text = re.sub(r'<[^>]+>', '', raw_html_text)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        with st.container(height=600):
            st.image(image, use_container_width=True)
            
    with col2:
        st.text_area("Extracted Text (Read-Only):", clean_text, height=600, disabled=True)
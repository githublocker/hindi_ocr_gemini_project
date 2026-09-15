import streamlit as st
from PIL import Image
import re
import os
import csv
import time
from surya.recognition import RecognitionPredictor
from surya.inference import SuryaInferenceManager

# Maximize the page width for side-by-side layout
st.set_page_config(layout="wide")

# Initialize Offline Surya OCR
@st.cache_resource
def load_surya_models():
    manager = SuryaInferenceManager()
    recognizer = RecognitionPredictor(manager)
    return recognizer

recognizer = load_surya_models()

# --- Floating Dataset Counter Logic ---
csv_path = "dataset/labels.csv"
dataset_count = 0
if os.path.isfile(csv_path):
    with open(csv_path, 'r', encoding='utf-8') as f:
        dataset_count = max(0, sum(1 for row in f) - 1)

if dataset_count >= 100:
    bg_color = "#28a745" # Green
    msg = f"✅ Ready to Train! (Saved: {dataset_count})"
else:
    bg_color = "#007bff" # Blue
    msg = f"💾 Saved: {dataset_count} / 100"

float_css = f"""
<style>
.floating-counter {{
    position: fixed;
    bottom: 30px;
    right: 30px;
    background-color: {bg_color};
    color: white;
    padding: 15px 25px;
    border-radius: 30px;
    z-index: 9999;
    font-family: sans-serif;
    font-weight: bold;
    box-shadow: 0px 4px 10px rgba(0,0,0,0.3);
}}
</style>
<div class="floating-counter">{msg}</div>
"""
st.markdown(float_css, unsafe_allow_html=True)
# --------------------------------------

st.title("Abhishek_Hindi_English_OCR")
uploaded_file = st.file_uploader("Upload Image")

if uploaded_file:
    image = Image.open(uploaded_file)
    
    # Offline Prediction
    predictions = recognizer([image])
    
    # Extract text and remove HTML tags
    raw_html_text = " ".join([block.html for block in predictions[0].blocks if block.html])
    clean_text = re.sub(r'<[^>]+>', '', raw_html_text)
    
    # Create a side-by-side layout
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.write("### Uploaded Image")
        with st.container(height=600):
            st.image(image, use_container_width=True)
            
    with col2:
        st.write("### Extracted Text")
        correct_text = st.text_area("Is this correct? Fix it below:", clean_text, height=520)
        
        if st.button("Save to Dataset"):
            os.makedirs("dataset/images", exist_ok=True)
            
            file_id = str(int(time.time()))
            image_path = f"dataset/images/{file_id}.jpg"
            
            image.save(image_path)
            
            file_exists = os.path.isfile(csv_path)
            
            with open(csv_path, mode='a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow(["image_path", "text"])
                writer.writerow([image_path, correct_text])
                
            st.success(f"Saved successfully to {csv_path}!")
            time.sleep(1) 
            st.rerun()
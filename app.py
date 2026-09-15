import streamlit as st
import pandas as pd
import easyocr

# Initialize Offline Hindi OCR
@st.cache_resource
def load_ocr():
    return easyocr.Reader(['hi', 'en'])
reader = load_ocr()

st.title("Offline Hindi OCR & Teacher Loop")
uploaded_file = st.file_uploader("Upload Image")

if uploaded_file:
    # Offline Prediction
    result = reader.readtext(uploaded_file.read(), detail=0)
    st.write("Offline OCR Prediction:", " ".join(result))
    
    # Manual Correction & Saving Loop
    st.write("Is this correct? If not, fix it below:")
    correct_text = st.text_input("Ground Truth Text", " ".join(result))
    
    if st.button("Save to Dataset"):
        # Add your local save logic here
        st.success("Saved for future fine-tuning!")

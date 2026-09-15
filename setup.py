from setuptools import setup

setup(
    name="Abhishek_Hindi_English_OCR",
    version="0.1.0",
    py_modules=["app", "1_Quick_Extract"],
    install_requires=[
        "streamlit",
        "surya-ocr",
        "pillow"
    ],
)
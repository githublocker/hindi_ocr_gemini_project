@echo off
echo Setting up Offline Hindi OCR...
python -m venv venv
call venv\Scripts\activate.bat
pip install -r requirements.txt
echo Setup Complete! Starting App...
streamlit run app.py
pause

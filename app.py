import streamlit as st
import joblib
import numpy as np
import os

# Set page title
st.set_page_config(page_title="Motor Health Monitor")

st.title("⚙️ Motor Failure Predictor")
st.write("Enter the motor sensor readings to predict health status.")

# 1. Load the trained model
MODEL_PATH = "models/motor_model.pkl"

if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
    
    # 2. Create Input Fields
    col1, col2 = st.columns(2)
    
    with col1:
        voltage = st.number_input("Voltage (V)", min_value=150.0, max_value=300.0, value=230.0)
        current = st.number_input("Current (A)", min_value=0.0, max_value=20.0, value=5.0)
        
    with col2:
        temp = st.number_input("Temperature (°C)", min_value=0.0, max_value=150.0, value=50.0)
        vib = st.number_input("Vibration (G)", min_value=0.0, max_value=1.0, value=0.05, format="%.3f")

    # 3. Prediction Logic
    if st.button("Check Motor Status", type="primary"):
        # Arrange features exactly as they were during training
        features = np.array([[voltage, current, temp, vib]])
        prediction = model.predict(features)
        probability = model.predict_proba(features)[0][1] # Chance of failure

        st.divider()
        
        if prediction[0] == 1:
            st.error(f"🚨 WARNING: Potential Failure Detected!")
            st.write(f"Confidence: {probability*100:.1f}%")
        else:
            st.success(f"✅ Motor Operating Normally.")
            st.write(f"Risk Level: {probability*100:.1f}%")
else:
    st.warning("⚠️ Model file not found! Please run 'python3 src/train.py' first.")

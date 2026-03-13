import streamlit as st
import joblib
import numpy as np
import pandas as pd
import os
from datetime import datetime
from src.monitor import check_data_drift
import plotly.graph_objects as go

# ============================================================================
# PAGE CONFIGURATION & STYLING
# ============================================================================
st.set_page_config(
    page_title="Motor Health Monitor | Industrial Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Professional Styling
custom_css = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }
    
    /* Main background and text */
    body {
        background-color: #0f1419;
        color: #e1e8ed;
    }
    
    /* Title styling */
    h1 {
        color: #ffffff;
        font-weight: 700;
        letter-spacing: -0.5px;
        margin-bottom: 8px;
    }
    
    h2 {
        color: #ffffff;
        font-weight: 600;
        margin-top: 24px;
    }
    
    h3 {
        color: #a1aab8;
        font-weight: 500;
        font-size: 14px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Input container styling */
    .metric-card {
        background: linear-gradient(135deg, #1a202c 0%, #2d3748 100%);
        border-radius: 12px;
        padding: 20px;
        border-left: 4px solid #00d4ff;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        margin-bottom: 16px;
    }
    
    .metric-card-warning {
        border-left-color: #ff6b6b;
    }
    
    .metric-card-success {
        border-left-color: #2ecc71;
    }
    
    /* Button styling with gradient and hover effect */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 12px 32px;
        font-weight: 600;
        font-size: 16px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
        width: 100%;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
    }
    
    .stButton > button:active {
        transform: translateY(0px);
    }
    
    /* Slider styling */
    .stSlider {
        padding: 12px 0;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a202c 0%, #2d3748 100%);
    }
    
    [data-testid="stSidebar"] > div > div:first-child {
        background: transparent;
    }
    
    /* Status box styling */
    .status-success {
        background: linear-gradient(135deg, rgba(46, 204, 113, 0.1) 0%, rgba(46, 204, 113, 0.05) 100%);
        border: 1px solid rgba(46, 204, 113, 0.3);
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 0 20px rgba(46, 204, 113, 0.1);
    }
    
    .status-failure {
        background: linear-gradient(135deg, rgba(220, 53, 69, 0.1) 0%, rgba(220, 53, 69, 0.05) 100%);
        border: 1px solid rgba(220, 53, 69, 0.3);
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 0 20px rgba(220, 53, 69, 0.1);
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0%, 100% {
            box-shadow: 0 0 20px rgba(220, 53, 69, 0.1);
        }
        50% {
            box-shadow: 0 0 30px rgba(220, 53, 69, 0.3);
        }
    }
    
    /* Divider styling */
    hr {
        border-color: rgba(255, 255, 255, 0.1);
        margin: 24px 0;
    }
    
    /* Container styling */
    .stContainer {
        background: rgba(255, 255, 255, 0.01);
        border-radius: 12px;
        padding: 20px;
    }
    
    /* Metric styling */
    [data-testid="metric-container"] {
        background: linear-gradient(135deg, #1a202c 0%, #2d3748 100%);
        border-radius: 12px;
        padding: 16px;
        border: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    /* Label styling */
    label {
        color: #a1aab8;
        font-weight: 500;
        font-size: 13px;
    }
    
    /* Warning and error box styling */
    .stAlert {
        border-radius: 12px;
        padding: 16px;
    }
</style>
"""

st.markdown(custom_css, unsafe_allow_html=True)

# ============================================================================
# CACHE MODEL LOADING
# ============================================================================
@st.cache_resource
def load_trained_model():
    """Load the trained model with caching to avoid reloading on every interaction."""
    MODEL_PATH = "models/motor_model.pkl"
    if not os.path.exists(MODEL_PATH):
        return None
    return joblib.load(MODEL_PATH)

# ============================================================================
# SIDEBAR - SYSTEM HEALTH & MODEL INFO
# ============================================================================
with st.sidebar:
    st.markdown("### SYSTEM CONTROL CENTER")
    st.divider()
    
    st.subheader("CURRENT SYSTEM STATUS")
    
    # Dummy system metrics (in production, these would be from a DB)
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Fleet Health", "94%", "↑ 3%", delta_color="off")
    with col2:
        st.metric("Active Units", 156, "↑ 12")
    
    st.divider()
    
    st.subheader("MODEL INFORMATION")
    st.markdown("""
    **Model Type:** Logistic Regression  
    **Accuracy:** 85.00%  
    **Precision:** 92%  
    **Recall:** 70%  
    **Training Samples:** 100  
    
    **Last Updated:** Today  
    """)
    
    st.divider()
    
    st.subheader("FEATURE RANGES")
    st.markdown("""
    - **Voltage:** 180-260V (nominal: 230V)
    - **Current:** 0-20A (nominal: 5A)
    - **Temperature:** 20-120°C (normal: 50°C)
    - **Vibration:** 0-1.0G (safe: <0.2G)
    """)
    
    st.divider()
    
    st.markdown("""
    <div style="text-align: center; color: #a1aab8; font-size: 12px; margin-top: 32px;">
        <p>Motor Maintenance MLOps v1.0</p>
        <p>Powered by Scikit-learn & Streamlit</p>
    </div>
    """, unsafe_allow_html=True)

# ============================================================================
# MAIN DASHBOARD
# ============================================================================
st.markdown("## MOTOR FAILURE PREDICTOR")
st.markdown("Industrial-grade predictive maintenance system with real-time diagnostic alerts")

# Top-level Status Bar
status_col1, status_col2, status_col3 = st.columns(3)
with status_col1:
    st.metric("FLEET HEALTH", "94%", "↑ 3%")
with status_col2:
    st.metric("ACTIVE UNITS", "156", "↑ 12")
with status_col3:
    st.metric("MODEL ACCURACY", "85.00%", "Target: >80%")

st.divider()

model = load_trained_model()

if model is None:
    st.error("Model file not found!")
    st.warning("Please run `python src/train.py` to generate the model first.")
else:
    # Sensor Input Module
    with st.container():
        st.markdown("### SENSOR INPUT DASHBOARD")
        st.markdown("---")
        
        # Two-column grid for inputs
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("**ELECTRICAL PARAMETERS**")
            voltage = st.slider(
                "Voltage (V)",
                min_value=180.0,
                max_value=260.0,
                value=230.0,
                step=0.5,
                help="Nominal: 230V",
                key="voltage_slider"
            )
            
            current = st.slider(
                "Current (A)",
                min_value=0.0,
                max_value=20.0,
                value=5.0,
                step=0.1,
                help="Nominal: 5A",
                key="current_slider"
            )
        
        with col2:
            st.markdown("**ENVIRONMENTAL PARAMETERS**")
            temp = st.slider(
                "Temperature (°C)",
                min_value=20.0,
                max_value=120.0,
                value=50.0,
                step=0.5,
                help="Normal: <80°C",
                key="temp_slider"
            )
            
            vib = st.slider(
                "Vibration (G)",
                min_value=0.0,
                max_value=1.0,
                value=0.05,
                step=0.01,
                help="Safe: <0.2G",
                key="vib_slider"
            )
        
        st.divider()
        
        # Prediction Button
        predict_button = st.button("ANALYZE MOTOR STATUS", key="predict_btn")
    
    # Prediction Results Section
    if predict_button:
        # Prepare features
        features = pd.DataFrame([[voltage, current, temp, vib]], columns=['voltage', 'current', 'temperature', 'vibration'])
        prediction = model.predict(features)
        probability = model.predict_proba(features)[0][1]  # Failure probability
        health_score = 100 - (probability * 100)
        
        # Check for data drift
        try:
            baseline_df = pd.read_csv('data/motor_data.csv')
            drift_warnings = check_data_drift(baseline_df, features)
        except FileNotFoundError:
            drift_warnings = []
            st.warning("Baseline data not found - cannot check for data drift")
        
        # Diagnostic Results Module
        with st.container():
            st.markdown("### DIAGNOSTIC RESULTS")
            st.markdown("---")
            
            # Result metrics
            result_col1, result_col2, result_col3 = st.columns(3)
            
            with result_col1:
                st.metric(
                    "MOTOR HEALTH SCORE",
                    f"{health_score:.1f}%",
                    delta=f"{'+' if health_score > 70 else ''}{health_score - 70:.1f}% vs baseline",
                    delta_color="off"
                )
            
            with result_col2:
                # Plotly Gauge for Failure Risk
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=probability * 100,
                    title={'text': "FAILURE RISK"},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 40], 'color': "green"},
                            {'range': [40, 75], 'color': "orange"},
                            {'range': [75, 100], 'color': "crimson"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 75
                        }
                    }
                ))
                fig.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    font={'color': "white", 'family': "Arial"}
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with result_col3:
                status_text = "HEALTHY" if prediction[0] == 0 else "AT RISK"
                st.metric("STATUS", status_text, delta_color="off")
            
            # Detailed Status Box
            if prediction[0] == 0:
                st.success("MOTOR OPERATING NORMALLY - All sensor readings within acceptable ranges. Continue routine monitoring.")
            else:
                st.error("WARNING: POTENTIAL FAILURE DETECTED - Anomalous sensor readings detected. Recommend immediate maintenance inspection.")
        
        # Data Drift Warnings
        if drift_warnings:
            st.markdown("### DATA DRIFT WARNING")
            st.markdown("The following sensor readings are statistical outliers compared to the training data baseline:")
            for warning in drift_warnings:
                st.warning(warning)
        
        st.divider()
        
        # Feature Importance Chart
        st.markdown("### FEATURE IMPORTANCE")
        st.markdown("This chart shows which sensors contributed most to the failure risk prediction:")
        
        # Calculate feature importance from model coefficients
        importance_values = abs(model.coef_[0])
        feature_names = ['Voltage', 'Current', 'Temperature', 'Vibration']
        
        # Create DataFrame for bar chart
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importance_values
        }).set_index('Feature')
        
        st.bar_chart(importance_df)
        
        st.divider()
        
        # Parameter Analysis Table
        st.markdown("### SENSOR PARAMETER ANALYSIS")
        
        analysis_data = {
            "Parameter": ["Voltage", "Current", "Temperature", "Vibration"],
            "Reading": [f"{voltage:.2f}V", f"{current:.2f}A", f"{temp:.2f}°C", f"{vib:.3f}G"],
            "Status": [
                "Normal" if 180 <= voltage <= 260 else "Caution",
                "Normal" if 0 <= current <= 20 else "Critical",
                "Normal" if temp < 80 else "Elevated" if temp < 100 else "Critical",
                "Normal" if vib < 0.2 else "Caution" if vib < 0.5 else "High",
            ]
        }
        
        st.dataframe(analysis_data, width='stretch', hide_index=True)
        
        # Recommendations
        st.markdown("### RECOMMENDATIONS")
        
        recommendations = []
        if voltage < 180 or voltage > 260:
            recommendations.append("Check power supply — voltage out of specification")
        if current > 15:
            recommendations.append("Verify load — current exceeding nominal range")
        if temp > 100:
            recommendations.append("Inspect cooling system — temperature critically high")
        if vib > 0.3:
            recommendations.append("Schedule maintenance — vibration levels elevated")
        
        if not recommendations:
            st.success("No urgent recommendations — system operating optimally")
        else:
            for rec in recommendations:
                st.warning(rec)
        
        # Timestamp
        st.markdown(
            f"""
            <div style="text-align: center; color: #7a8a99; font-size: 12px; margin-top: 24px;">
            <p>Analysis completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            """,
            unsafe_allow_html=True
        )

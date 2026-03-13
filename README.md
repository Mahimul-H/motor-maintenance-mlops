# Motor Maintenance MLOps Pipeline

[![ML Pipeline Test](https://github.com/Mahimul-H/motor-maintenance-mlops/actions/workflows/ml_pipeline.yml/badge.svg)](https://github.com/Mahimul-H/motor-maintenance-mlops/actions)

## Overview
This repository contains an end-to-end Machine Learning Operations (MLOps) pipeline for predictive maintenance. The system utilizes simulated sensor telemetry to predict motor failure, demonstrating the integration of data engineering, model training, containerization, and automated CI/CD workflows.

## Technical Stack
* **Language:** Python 3.10
* **Machine Learning:** Scikit-learn (Random Forest Classifier)
* **Web Framework:** Streamlit
* **Containerization:** Docker
* **Automation:** GitHub Actions

## System Architecture
1. **Data Layer:** Synthetic generation of sensor telemetry (Voltage, Current, Temperature, Vibration).
2. **Modeling Layer:** Training and serialization of a classification model for binary failure prediction.
3. **Deployment Layer:** A containerized Streamlit application for real-time inference.
4. **Operations Layer:** GitHub Actions workflow for automated model testing and environment validation.

## Model Card

### Model Details
- **Model Type:** Logistic Regression (binary classifier)
- **Task:** Binary classification for motor failure prediction
- **Features Used:**
  - **Voltage:** Electrical voltage measurement (V) - nominal range: 180-260V
  - **Current:** Electrical current measurement (A) - nominal range: 0-20A
  - **Temperature:** Motor temperature (°C) - safe range: <80°C
  - **Vibration:** Vibration amplitude (G-force) - safe range: <0.2G
- **Target Variable:** Binary failure indicator (0 = normal, 1 = failure)
- **Training Data:** 100 synthetic samples (80% train, 20% test split)
- **Performance Metrics:**
  - Accuracy: 85.00%
  - Precision: 92%
  - Recall: 70%

### Intended Use
This model is designed for industrial maintenance applications to predict motor failure based on real-time sensor telemetry. It provides early warning of potential motor issues, enabling preventive maintenance and reducing downtime in manufacturing and industrial settings. The model should be used as a decision-support tool by maintenance engineers and operators.

### Limitations
- **Synthetic Data Training:** The model was trained exclusively on synthetically generated data, which may not capture all real-world variability, noise, or edge cases present in actual industrial environments.
- **Limited Sample Size:** Training on only 100 samples may lead to overfitting and reduced generalization capability.
- **Feature Scope:** The model considers only four basic sensor measurements; real-world applications might require additional features like motor age, usage history, or environmental factors.
- **Binary Classification:** The model provides binary failure predictions without confidence intervals or multi-class failure modes.
- **Assumption of Stationarity:** The model assumes sensor data distributions remain consistent over time, which may not hold in dynamic industrial settings.

## Directory Structure
* /src: Model training and evaluation logic.
* /data_generator: Scripts for generating synthetic sensor data.
* /models: Serialized model artifacts (.pkl).
* /app.py: Streamlit application interface.
* Dockerfile: Configuration for containerized deployment.

## Installation and Deployment

### Docker Deployment (Recommended)
Build the image and execute the container:
```bash
docker build -t motor-maintenance-app .
docker run -p 8501:8501 motor-maintenance-app
```

### Manual Installation
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Execute the application:
   ```bash
   streamlit run app.py
   ```

## Continuous Integration
This project utilizes GitHub Actions to ensure code quality and model integrity. Every push to the main branch triggers an automated test suite that installs dependencies and executes the training pipeline to verify the build.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

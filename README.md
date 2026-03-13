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
add docker deployment for Docker as well 

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

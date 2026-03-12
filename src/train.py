import pandas as pd
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# 1. Setup paths
DATA_PATH = "data/motor_data.csv"
MODEL_DIR = "models"
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

def train_model():
    # 2. Load Data
    df = pd.read_csv(DATA_PATH)
    
    # Features (X) and Target (y)
    X = df[['voltage', 'current', 'temperature', 'vibration']]
    y = df['failure']
    
    # 3. Split data (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 4. Initialize and Train Model
    print("🚀 Training the Logistic Regression model...")
    model = LogisticRegression()
    model.fit(X_train, y_train)
    
    # 5. Evaluate
    predictions = model.predict(X_test)
    acc = accuracy_score(y_test, predictions)
    
    print(f"✅ Training Complete!")
    print(f"📊 Accuracy: {acc * 100:.2f}%")
    print("\nDetailed Report:")
    print(classification_report(y_test, predictions))
    
    # 6. Save the model
    model_path = os.path.join(MODEL_DIR, "motor_model.pkl")
    joblib.dump(model, model_path)
    print(f"💾 Model saved to: {model_path}")

if __name__ == "__main__":
    train_model()
import pandas as pd
import numpy as np
import os

# 1. Create the 'data' directory if it doesn't exist
if not os.path.exists('data'):
    os.makedirs('data')
    print("📁 Created 'data/' directory.")

# 2. Set seed for consistent results
np.random.seed(42)
n_rows = 100

# 3. Generate Features
# Sensors usually have some noise, so we use uniform and normal distributions
voltage = np.random.uniform(200, 240, n_rows)     # Volts
current = np.random.uniform(4.0, 10.0, n_rows)    # Amps
temperature = np.random.uniform(40, 110, n_rows)  # Celsius
vibration = np.random.uniform(0.01, 0.35, n_rows) # G-force

# 4. Define Failure Logic (The "Ground Truth")
# We make failure dependent on High Temp AND High Vibration
# Logic: If (Temp > 90 and Vib > 0.2) or (Temp > 105), it's likely a failure
failure_prob = (
    (temperature * 0.4) + 
    (vibration * 150) + 
    (current * 2) + 
    np.random.normal(0, 5, n_rows) # Adding some "noise" to make it realistic
)

# Threshold for binary classification (0 = Healthy, 1 = Fail)
failure = (failure_prob > 80).astype(int)

# 5. Create DataFrame
df = pd.DataFrame({
    'voltage': np.round(voltage, 2),
    'current': np.round(current, 2),
    'temperature': np.round(temperature, 2),
    'vibration': np.round(vibration, 3),
    'failure': failure
})

# 6. Save to CSV
file_path = 'data/motor_data.csv'
df.to_csv(file_path, index=False)

print(f"✅ Generated {n_rows} rows of motor data at: {file_path}")
print(f"📊 Failure breakdown: {df['failure'].value_counts().to_dict()}")

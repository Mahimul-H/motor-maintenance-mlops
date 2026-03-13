import pandas as pd
import numpy as np

def check_data_drift(baseline_df, current_df):
    """
    Check for data drift by comparing current sensor readings to baseline statistics.
    
    Args:
        baseline_df: DataFrame with historical sensor data
        current_df: DataFrame with current sensor readings (single row)
    
    Returns:
        List of warning messages for sensors that are statistical outliers
    """
    warnings = []
    
    # Sensor columns to check
    sensors = ['voltage', 'current', 'temperature', 'vibration']
    
    for sensor in sensors:
        if sensor in baseline_df.columns and sensor in current_df.columns:
            # Calculate baseline statistics
            baseline_mean = baseline_df[sensor].mean()
            baseline_std = baseline_df[sensor].std()
            
            # Get current value
            current_value = current_df[sensor].iloc[0]
            
            # Check if current value is more than 2 standard deviations from baseline mean
            deviation = abs(current_value - baseline_mean)
            threshold = 2 * baseline_std
            
            if deviation > threshold:
                direction = "high" if current_value > baseline_mean else "low"
                warnings.append(
                    f"**{sensor.title()} Drift Detected**: Current reading ({current_value:.2f}) is "
                    f"significantly {direction} compared to baseline "
                    f"(mean: {baseline_mean:.2f}, ±2σ: {baseline_mean - threshold:.2f} to {baseline_mean + threshold:.2f})"
                )
    
    return warnings
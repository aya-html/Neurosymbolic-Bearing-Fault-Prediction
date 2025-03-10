import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis

# Path to the selected features dataset
input_path = r'C:\Users\T14s\Desktop\my 1 research paper\selected_features.csv'
output_path = r'C:\Users\T14s\Desktop\my 1 research paper\enhanced_features.csv'

# Load the dataset
try:
    data = pd.read_csv(input_path)
    print("Selected Features CSV loaded successfully.")
except Exception as e:
    print(f"Error loading selected features CSV: {e}")
    exit()

# Verify the dataset
print("\nFirst few rows of the dataset:")
print(data.head())

# Compute additional statistical features
try:
    # Add std, skewness, and kurtosis (cragness) columns
    data['std'] = data['mean'].rolling(window=3, min_periods=1).std().fillna(0)
    data['skewness'] = data['mean'].rolling(window=3, min_periods=1).apply(skew).fillna(0)
    data['cragness'] = data['mean'].rolling(window=3, min_periods=1).apply(kurtosis).fillna(0)
    print("\nAdditional statistical features added.")
except Exception as e:
    print(f"Error computing additional features: {e}")
    exit()

# Save the enhanced dataset
try:
    data.to_csv(output_path, index=False)
    print(f"Enhanced dataset saved to: {output_path}")
except Exception as e:
    print(f"Error saving enhanced dataset: {e}")

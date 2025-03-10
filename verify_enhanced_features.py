import pandas as pd

# Path to the enhanced features dataset
enhanced_features_path = r'C:\Users\T14s\Desktop\my 1 research paper\enhanced_features.csv'

# Load the enhanced dataset
try:
    data = pd.read_csv(enhanced_features_path)
    print("Enhanced Features CSV loaded successfully.")
    print("\nFirst few rows of the dataset:")
    print(data.head())

    # Check for missing values and class distribution
    print("\nMissing values in each column:")
    print(data.isnull().sum())
except Exception as e:
    print(f"Error loading enhanced features CSV: {e}")

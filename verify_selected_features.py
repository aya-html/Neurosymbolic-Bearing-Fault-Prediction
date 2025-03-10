import pandas as pd

# Path to the selected features CSV
selected_features_path = r'C:\Users\T14s\Desktop\my 1 research paper\selected_features.csv'

# Load the selected features
data = pd.read_csv(selected_features_path)
print("Selected Features CSV loaded successfully.")

# Display summary of the data
print("\nSummary statistics:")
print(data.describe())

# Check for missing values
print("\nMissing values in each column:")
print(data.isnull().sum())

# Check class distribution
print("\nClass distribution:")
print(data['label'].value_counts())

import pandas as pd

# Path to the extracted features CSV
features_path = r'C:\Users\T14s\Desktop\my 1 research paper\extracted_features.csv'

# Load and inspect the extracted features
try:
    # Load the file
    features_data = pd.read_csv(features_path)
    print("Extracted Features CSV loaded successfully.")
    
    # Display the first few rows of the dataset
    print("\nFirst few rows of the extracted features:")
    print(features_data.head())

    # Display summary statistics
    print("\nSummary statistics:")
    print(features_data.describe())

    # Check for missing values
    missing_values = features_data.isnull().sum()
    print("\nMissing values in each column:")
    print(missing_values)
    
    # Check the distribution of the target label
    if 'label' in features_data.columns:
        print("\nClass distribution:")
        print(features_data['label'].value_counts())
    else:
        print("\nWarning: Target label column 'label' is missing.")
except Exception as e:
    print(f"Error loading or verifying extracted features: {e}")

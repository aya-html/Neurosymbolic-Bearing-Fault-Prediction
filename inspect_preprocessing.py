import pandas as pd

# Provide the path to your .pkl file
file_path = r"C:\Users\T14s\Desktop\my 1 research paper\dataset used in the chosen paper\dataframes_module\75Hz_2classes.pkl"

# Load the dataset
data = pd.read_pickle(file_path)

# Display dataset columns and sample data
print("Dataset Columns:")
print(data.columns)

print("\nSample Data:")
print(data.head())

# Check for specific preprocessing-related features
preprocessing_indicators = ['mean', 'kurtosis', 'fft', 'std', 'dc_offset', 'notch']
print("\nChecking Preprocessing Indicators in Column Names:")
for indicator in preprocessing_indicators:
    matching_columns = [col for col in data.columns if indicator in col.lower()]
    print(f"{indicator}: {matching_columns if matching_columns else 'Not Found'}")

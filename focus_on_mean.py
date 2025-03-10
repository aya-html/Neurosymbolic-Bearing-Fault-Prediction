import pandas as pd

# Path to the extracted features CSV
extracted_features_path = r'C:\Users\T14s\Desktop\my 1 research paper\extracted_features.csv'
selected_features_path = r'C:\Users\T14s\Desktop\my 1 research paper\selected_features.csv'

# Load the extracted features
data = pd.read_csv(extracted_features_path)
print("Extracted Features CSV loaded successfully.")

# Select only the 'mean' and 'label' columns
selected_data = data[['mean', 'label']]
print("\nSelected columns preview:")
print(selected_data.head())

# Save the selected features to a new CSV file
selected_data.to_csv(selected_features_path, index=False)
print(f"\nSelected features saved to: {selected_features_path}")

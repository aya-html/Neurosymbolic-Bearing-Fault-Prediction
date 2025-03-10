import pandas as pd

# Path to the preprocessed dataset (filtered)
dataset_path = r'C:\Users\T14s\Desktop\my 1 research paper\dataset used in the chosen paper\dataframes_module\filtered_dataset.pkl'

# Load the dataset
try:
    data = pd.read_pickle(dataset_path)
    print("Dataset loaded successfully.")
except Exception as e:
    print(f"Error loading dataset: {e}")
    exit()

# Select relevant features
relevant_columns = [
    'mean_10_25', 'std_class_10_25', 'skew_10_25', 'kurt_10_25',  # Time-domain features
    'mean_25_75', 'std_class_25_75', 'skew_25_75', 'kurt_25_75',  # Additional features
    'D_class'  # Target label
]

# Ensure selected columns are in the dataset
missing_columns = [col for col in relevant_columns if col not in data.columns]
if missing_columns:
    print(f"Error: Missing columns in dataset: {missing_columns}")
    exit()

# Create a new dataset with selected features
selected_features = data[relevant_columns]

# Save the selected features to a new file
output_path = r'C:\Users\T14s\Desktop\my 1 research paper\selected_features.csv'
try:
    selected_features.to_csv(output_path, index=False)
    print(f"Selected features saved to '{output_path}'.")
except Exception as e:
    print(f"Error saving selected features: {e}")

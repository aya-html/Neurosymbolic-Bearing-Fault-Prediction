import pandas as pd

# Path to the preprocessed dataset
dataset_path = r'C:\Users\T14s\Desktop\my 1 research paper\dataset used in the chosen paper\dataframes_module\filtered_dataset.pkl'

# Load dataset
try:
    data = pd.read_pickle(dataset_path)
    print("Dataset loaded successfully.")
    
    # Display the first few rows
    print("First few rows of the dataset:")
    print(data.head())

    # Check for other potential signal-like columns
    for col in data.columns:
        print(f"Column: {col}, Type: {type(data[col].iloc[0])}")
except Exception as e:
    print(f"Error loading dataset: {e}")

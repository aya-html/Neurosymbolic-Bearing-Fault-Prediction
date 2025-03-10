import pickle
import pandas as pd

# Path to the normalized dataset
output_file_path = r"C:\Users\T14s\Desktop\my 1 research paper\dataset used in the chosen paper\dataframes_module\normalized_dataset.pkl"

try:
    # Load the normalized dataset
    with open(output_file_path, 'rb') as file:
        normalized_data = pickle.load(file)

    # Check the type of data and inspect the first few rows
    if isinstance(normalized_data, pd.DataFrame):
        print("Normalized dataset loaded successfully.")
        print("\nFirst 5 rows of the normalized dataset:")
        print(normalized_data.head())
        print("\nSummary statistics:")
        print(normalized_data.describe())
    else:
        print(f"Unexpected data type: {type(normalized_data)}. Expected a Pandas DataFrame.")

except Exception as e:
    print(f"An error occurred while verifying the normalized dataset: {e}")

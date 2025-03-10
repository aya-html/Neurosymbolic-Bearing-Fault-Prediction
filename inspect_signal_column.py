import pandas as pd

# Path to the preprocessed dataset
dataset_path = r'C:\Users\T14s\Desktop\my 1 research paper\dataset used in the chosen paper\dataframes_module\filtered_dataset.pkl'

try:
    # Load the dataset
    data = pd.read_pickle(dataset_path)
    print("Dataset loaded successfully.")

    # Inspect the 'mean_10_25' column
    print("Sample of 'mean_10_25' column values:")
    print(data['mean_10_25'].head())
except KeyError:
    print("The column 'mean_10_25' does not exist.")
except Exception as e:
    print(f"An error occurred: {e}")

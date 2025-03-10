import pandas as pd
from sklearn.preprocessing import StandardScaler
import pickle
import os

# File paths
input_file_path = r"C:\Users\T14s\Desktop\my 1 research paper\dataset used in the chosen paper\dataframes_module\75Hz_2classes.pkl"
output_file_path = r"C:\Users\T14s\Desktop\my 1 research paper\dataset used in the chosen paper\dataframes_module\normalized_dataset.pkl"

try:
    # Load the dataset
    with open(input_file_path, 'rb') as file:
        data = pickle.load(file)

    # Ensure the dataset is a DataFrame
    if not isinstance(data, pd.DataFrame):
        raise ValueError("The loaded dataset is not a Pandas DataFrame.")

    print("Original dataset loaded successfully.")

    # Extract features and exclude non-feature columns
    feature_columns = data.columns.difference(['D_class', 'name_signal'])
    features = data[feature_columns]

    # Initialize the scaler
    scaler = StandardScaler()

    # Normalize features
    normalized_features = pd.DataFrame(
        scaler.fit_transform(features),
        columns=feature_columns
    )
    
    # Combine normalized features with the target and metadata
    normalized_dataset = pd.concat([data[['name_signal', 'D_class']].reset_index(drop=True),
                                     normalized_features.reset_index(drop=True)], axis=1)

    # Save the normalized dataset
    with open(output_file_path, 'wb') as output_file:
        pickle.dump(normalized_dataset, output_file)

    print(f"Normalization complete. Normalized dataset saved to: {output_file_path}")

except FileNotFoundError:
    print(f"Input file not found at: {input_file_path}")
except Exception as e:
    print(f"An error occurred: {e}")

import pandas as pd

# Paths to the split dataset files
file_paths = {
    "X_train": "X_train.csv",
    "y_train": "y_train.csv",
    "X_val": "X_val.csv",
    "y_val": "y_val.csv",
    "X_test": "X_test.csv",
    "y_test": "y_test.csv",
}

# Expected sizes for each split
expected_sizes = {
    "X_train": 378,
    "y_train": 378,
    "X_val": 81,
    "y_val": 81,
    "X_test": 81,
    "y_test": 81,
}

# Verification process
try:
    for key, path in file_paths.items():
        # Load the CSV file
        data = pd.read_csv(path)
        print(f"{key} loaded successfully. Shape: {data.shape}")

        # Check the number of rows
        if data.shape[0] != expected_sizes[key]:
            print(f"Warning: {key} has {data.shape[0]} rows, expected {expected_sizes[key]}.")

        # Check feature consistency
        if "X" in key:
            print(f"{key} features: {list(data.columns)}")
        elif "y" in key:
            class_distribution = data.iloc[:, 0].value_counts()
            print(f"{key} class distribution:\n{class_distribution}")

    print("\nVerification complete. All files loaded successfully.")
except Exception as e:
    print(f"An error occurred during verification: {e}")

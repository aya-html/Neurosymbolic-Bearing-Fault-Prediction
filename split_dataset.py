import pandas as pd
from sklearn.model_selection import train_test_split

# Path to the selected features dataset
selected_features_path = r'C:\Users\T14s\Desktop\my 1 research paper\selected_features.csv'

# Load the selected features dataset
try:
    data = pd.read_csv(selected_features_path)
    print("Selected Features CSV loaded successfully.")
except Exception as e:
    print(f"Error loading selected features CSV: {e}")
    exit()

# Separate features and labels
X = data.drop(columns=['label'])
y = data['label']

# Split the dataset into training (70%) and temp (30%)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

# Split the temp set into validation (15%) and testing (15%)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

# Verify the splits
print(f"Training set: {X_train.shape[0]} samples")
print(f"Validation set: {X_val.shape[0]} samples")
print(f"Testing set: {X_test.shape[0]} samples")

# Save the datasets to CSV files
X_train.to_csv('X_train.csv', index=False)
X_val.to_csv('X_val.csv', index=False)
X_test.to_csv('X_test.csv', index=False)
y_train.to_csv('y_train.csv', index=False)
y_val.to_csv('y_val.csv', index=False)
y_test.to_csv('y_test.csv', index=False)

print("Data splitting complete. Files saved:")
print("X_train.csv, y_train.csv, X_val.csv, y_val.csv, X_test.csv, y_test.csv")

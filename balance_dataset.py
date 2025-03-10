from imblearn.over_sampling import SMOTE
import pandas as pd

# Path to the selected features CSV
selected_features_path = r'C:\Users\T14s\Desktop\my 1 research paper\selected_features.csv'
balanced_features_path = r'C:\Users\T14s\Desktop\my 1 research paper\balanced_features.csv'

# Load the selected features
data = pd.read_csv(selected_features_path)
print("Selected Features CSV loaded successfully.")

# Split features and labels
X = data[['mean']]
y = data['label']

# Apply SMOTE for oversampling
smote = SMOTE(random_state=42)
X_balanced, y_balanced = smote.fit_resample(X, y)

# Create a balanced dataset
balanced_data = pd.DataFrame(X_balanced, columns=['mean'])
balanced_data['label'] = y_balanced

# Save the balanced dataset
balanced_data.to_csv(balanced_features_path, index=False)
print(f"\nBalanced dataset saved to: {balanced_features_path}")

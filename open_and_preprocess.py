import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree


import pickle

# Provide the path to my .pkl file
file_path = r"C:\Users\T14s\Desktop\my 1 research paper\dataset used in the chosen paper\dataframes_module\75Hz_2classes.pkl"

try:
    # Load the .pkl file
    with open(file_path, 'rb') as file:
        data = pickle.load(file)

    # Inspect the structure of the data
    print(f"Data type: {type(data)}")
    if isinstance(data, dict):
        print(f"Keys: {list(data.keys())}")
        for key, value in data.items():
            print(f"{key}: {type(value)} with shape {len(value) if hasattr(value, '__len__') else 'N/A'}")
    else:
        print("Content:", data)

except Exception as e:
    print(f"An error occurred: {e}")

# Load features and target labels
features = data.drop(columns=['D_class', 'name_signal'])  # Exclude label and non-feature columns
labels = data['D_class']

# Print sample data
print("\nFeatures sample:")
print(features.head())
print("\nLabels sample:")
print(labels.value_counts())

# Split data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

print(f"\nTraining samples: {X_train.shape[0]}, Testing samples: {X_test.shape[0]}")

# Train Decision Tree model
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

# Make predictions
y_pred = clf.predict(X_test)

# Evaluate the model
print("\nModel Evaluation:")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Plot the Decision Tree
plt.figure(figsize=(20, 10))
plot_tree(clf, filled=True, feature_names=features.columns, class_names=['Healthy', 'Fault'])
plt.title("Decision Tree Classifier")
plt.show()

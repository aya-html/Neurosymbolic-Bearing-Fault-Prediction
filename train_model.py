import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import tensorflow as tf

# ---------------------------
# Load and Preprocess Dataset
# ---------------------------
data = pd.read_csv('C:/Users/T14s/Desktop/my 1 research paper/enhanced_features.csv')

# Convert any negative label (e.g., -1, -2) to 1; leave 0 unchanged.
data['label'] = data['label'].apply(lambda x: 1 if x < 0 else x)

# Use LabelEncoder to ensure labels are 0 and 1
encoder = LabelEncoder()
data['label'] = encoder.fit_transform(data['label'])
print("Unique labels after encoding:", np.unique(data['label']))

# Check class distribution before filtering
class_counts = data['label'].value_counts()
print("Class Distribution Before Filtering:\n", class_counts)

# Filter out classes with fewer than 2 samples (if any)
data = data.groupby('label').filter(lambda x: len(x) > 1)
class_counts_filtered = data['label'].value_counts()
print("Class Distribution After Filtering:\n", class_counts_filtered)

# ---------------------------
# Select Features and Target Variable
# ---------------------------
feature_columns = ['mean', 'std', 'skewness', 'cragness']  # adjust if needed
X = data[feature_columns]
y = data['label'].astype(int)

# ---------------------------
# Apply SMOTE (if possible)
# ---------------------------
min_class_count = class_counts_filtered.min()
if min_class_count < 2:
    print("⚠ Warning: One or more classes have less than 2 samples. Skipping SMOTE.")
    X_smote, y_smote = X, y
    class_counts_after_smote = y.value_counts()
else:
    # Set k_neighbors=1 to avoid the error when the minority class has only one sample.
    k_neighbors_value = 1
    try:
        smote = SMOTE(random_state=42, k_neighbors=k_neighbors_value)
        X_smote, y_smote = smote.fit_resample(X, y)
        class_counts_after_smote = pd.Series(y_smote).value_counts()
    except ValueError as e:
        print(f"⚠ SMOTE Error: {e}")
        print("Skipping SMOTE due to class size issue.")
        X_smote, y_smote = X, y
        class_counts_after_smote = y.value_counts()

# Visualize class distribution after SMOTE
plt.figure(figsize=(6, 4))
colors = ['green', 'red'][:len(class_counts_after_smote)]
class_counts_after_smote.plot(kind='bar', color=colors)
plt.title('Class Distribution After SMOTE')
plt.xlabel('Class')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# ---------------------------
# Data Splitting and Scaling
# ---------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_smote, y_smote, test_size=0.2, random_state=42, stratify=y_smote
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------------------------
# Compute Class Weights for Models That Support Them
# ---------------------------
unique_classes = np.unique(y_train)
class_weights = compute_class_weight("balanced", classes=unique_classes, y=y_train)
class_weight_dict = {cls: weight for cls, weight in zip(unique_classes, class_weights)}
print("Class weights:", class_weight_dict)

# ---------------------------
# Model Training and Evaluation Functions
# ---------------------------
results = {}
probabilities = {}

def evaluate_model(model, X_eval, y_eval, model_name):
    predictions = model.predict(X_eval)
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_eval)[:, 1]
    else:
        proba = None

    acc = accuracy_score(y_eval, predictions)
    cm = confusion_matrix(y_eval, predictions)
    roc_auc = roc_auc_score(y_eval, proba) if proba is not None else None

    results[model_name] = {
        'Accuracy': acc,
        'Confusion Matrix': cm,
        'ROC AUC': roc_auc,
        'Classification Report': classification_report(y_eval, predictions, zero_division=0)
    }
    
    print(f"\n{model_name} Results:")
    print(f"Accuracy: {acc:.4f}")
    if roc_auc is not None:
        print(f"ROC AUC: {roc_auc:.4f}")
    print("Confusion Matrix:\n", cm)
    print("Classification Report:\n", results[model_name]['Classification Report'])
    
    return proba

# ---------------------------
# Train and Evaluate Models
# ---------------------------
# 1. Decision Tree (with class weights)
dt_model = DecisionTreeClassifier(class_weight=class_weight_dict, random_state=42)
dt_model.fit(X_train_scaled, y_train)
probabilities['Decision Tree'] = evaluate_model(dt_model, X_test_scaled, y_test, "Decision Tree")

# 2. Random Forest (with class weights)
rf_model = RandomForestClassifier(class_weight=class_weight_dict, n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)
probabilities['Random Forest'] = evaluate_model(rf_model, X_test_scaled, y_test, "Random Forest")

# 3. Gradient Boosting (class_weight not supported)
gb_model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
gb_model.fit(X_train_scaled, y_train)
probabilities['Gradient Boosting'] = evaluate_model(gb_model, X_test_scaled, y_test, "Gradient Boosting")

# 4. Neural Network (CNN)
cnn_model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X_train_scaled.shape[1],)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(1, activation='sigmoid')  # Binary classification
])
cnn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
cnn_model.fit(X_train_scaled, y_train, epochs=10, batch_size=32, verbose=1)
cnn_predictions = (cnn_model.predict(X_test_scaled) > 0.5).astype("int32")
cnn_roc_auc = roc_auc_score(y_test, cnn_model.predict(X_test_scaled))
results['Neural Network (CNN)'] = {
    'Accuracy': accuracy_score(y_test, cnn_predictions),
    'Confusion Matrix': confusion_matrix(y_test, cnn_predictions),
    'ROC AUC': cnn_roc_auc,
    'Classification Report': classification_report(y_test, cnn_predictions, zero_division=0)
}
probabilities['Neural Network (CNN)'] = cnn_model.predict(X_test_scaled).flatten()

# ---------------------------
# Plot ROC Curves for All Models
# ---------------------------
plt.figure(figsize=(8, 6))
for model_name, proba in probabilities.items():
    if proba is not None:
        fpr, tpr, _ = roc_curve(y_test, proba, pos_label=1)
        plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc_score(y_test, proba):.2f})')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Comparison')
plt.legend()
plt.grid(alpha=0.5)
plt.savefig('Figure_5_ROC_All_Models.png')
plt.show()

# ---------------------------
# Save Results to a Text File
# ---------------------------
with open('Results/Step_4_Advanced_Models_Comparison.txt', 'w') as f:
    for model_name, metrics in results.items():
        f.write(f'{model_name}:\n')
        for metric, value in metrics.items():
            f.write(f'  {metric}: {value}\n')
        f.write('\n')

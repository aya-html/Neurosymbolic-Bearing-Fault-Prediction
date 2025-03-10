import pandas as pd
import pickle
from scipy.signal import iirnotch, filtfilt
import numpy as np
import matplotlib.pyplot as plt

# Load the normalized dataset
input_path = r"C:\Users\T14s\Desktop\my 1 research paper\dataset used in the chosen paper\dataframes_module\normalized_dataset.pkl"
output_path = r"C:\Users\T14s\Desktop\my 1 research paper\dataset used in the chosen paper\dataframes_module\filtered_dataset.pkl"

try:
    with open(input_path, 'rb') as file:
        data = pickle.load(file)
    print("Normalized dataset loaded successfully.")
except Exception as e:
    print(f"Error loading dataset: {e}")
    exit()

# Define notch filter function
def apply_notch_filter(signal, fs, freqs, quality_factor=30):
    for freq in freqs:
        w0 = freq / (fs / 2)  # Normalize frequency
        b, a = iirnotch(w0, quality_factor)
        signal = filtfilt(b, a, signal)
    return signal

# Parameters
fs = 25600  # Sampling frequency (from chosen paper)
notch_freqs = np.arange(50, 550, 50)  # 50 Hz to 500 Hz in steps of 50 Hz

# Apply notch filtering to all features
features = data.drop(columns=['name_signal', 'D_class'])
filtered_features = features.apply(lambda col: apply_notch_filter(col, fs, notch_freqs) if np.issubdtype(col.dtype, np.number) else col, axis=0)

# Add labels and signal names back
filtered_features['D_class'] = data['D_class']
filtered_features['name_signal'] = data['name_signal']

# Save the filtered dataset
with open(output_path, 'wb') as file:
    pickle.dump(filtered_features, file)
print(f"Filtered dataset saved to: {output_path}")

# Plot before and after filtering for a sample signal
sample_signal = features.iloc[0].values
filtered_signal = apply_notch_filter(sample_signal, fs, notch_freqs)

# Frequency spectrum before filtering
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.magnitude_spectrum(sample_signal, Fs=fs, scale='dB')
plt.title("Frequency Spectrum Before Filtering")

# Frequency spectrum after filtering
plt.subplot(2, 1, 2)
plt.magnitude_spectrum(filtered_signal, Fs=fs, scale='dB')
plt.title("Frequency Spectrum After Filtering")

plt.tight_layout()
plt.show()

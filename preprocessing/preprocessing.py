# TODO: Implement this module
import os
import numpy as np
import scipy.io
import scipy.signal as signal
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

# --- Config ---
TARGET_FS = 256  # Unified input length
FILTER_BAND = (0.5, 40)

# --- Preprocessing Utilities ---
def bandpass_filter(eeg_signal, fs, low=0.5, high=40.0, order=5):
    nyq = 0.5 * fs
    lowcut = low / nyq
    highcut = high / nyq
    b, a = signal.butter(order, [lowcut, highcut], btype='band')
    return signal.filtfilt(b, a, eeg_signal)

def normalize_epoch(epoch):
    return (epoch - np.mean(epoch)) / np.std(epoch)

def pad_or_trim(epoch, target_len=256):
    if len(epoch) > target_len:
        return epoch[:target_len]
    elif len(epoch) < target_len:
        return np.pad(epoch, (0, target_len - len(epoch)))
    return epoch

def load_bonn_data(paths_dict, label_map, fs):
    data = []
    labels = []
    for class_name, path in paths_dict.items():
        for file in os.listdir(path):
            if file.endswith('.txt'):
                signal_data = np.loadtxt(os.path.join(path, file))
                filtered = bandpass_filter(signal_data, fs)
                normed = normalize_epoch(filtered)
                padded = pad_or_trim(normed, TARGET_FS)
                data.append(padded)
                labels.append(label_map[class_name])
    return np.array(data), np.array(labels)

def load_hauz_data(paths_dict, label_map, fs):
    data = []
    labels = []
    for class_name, path in paths_dict.items():
        for file in os.listdir(path):
            if file.endswith('.mat'):
                mat = scipy.io.loadmat(os.path.join(path, file))
                # Expecting 'data' in .mat file
                raw_signal = mat.get('data')
                if raw_signal is None:
                    continue
                signal_data = raw_signal.squeeze()
                filtered = bandpass_filter(signal_data, fs)
                normed = normalize_epoch(filtered)
                padded = pad_or_trim(normed, TARGET_FS)
                data.append(padded)
                labels.append(label_map[class_name])
    return np.array(data), np.array(labels)

def apply_rfe(X, y, num_features=64):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = LogisticRegression(max_iter=500)
    selector = RFE(model, n_features_to_select=num_features)
    X_reduced = selector.fit_transform(X_scaled, y)
    return X_reduced

# Example usage (in notebook/script):
# from preprocessing import load_bonn_data, load_hauz_data
# data, labels = load_bonn_data(DATASET_PATHS["bonn"], LABEL_MAPPINGS["bonn_s_z"], fs=173.61)
# or
# data, labels = load_hauz_data(DATASET_PATHS["hauz"], LABEL_MAPPINGS["hauz_multi_hauz"], fs=250.0)
# Optionally: data = apply_rfe(data, labels)

# This ensures output is (N, 256) ready for time-binning or SNN input

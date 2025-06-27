import os
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import torch
from src.utils.preprocessing import load_and_preprocess_bonn, load_and_preprocess_hauz
from config import DATASET_PATHS, LABEL_MAPPINGS, SAMPLING_FREQ

class EEGDataset(Dataset):
    def __init__(self, data, labels, T):
        self.data = data
        self.labels = labels
        self.T = T

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = self.data[idx]  # [features]
        x = np.tile(x, (self.T, 1))  # Repeat features across T time steps → [T, features]
        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y

def get_dataset(task_name, dataset_name, T=28, test_size=0.2, val_size=0.1, random_state=42):
    label_map = LABEL_MAPPINGS[task_name]
    raw_data = []
    raw_labels = []

    if dataset_name == "bonn":
        for class_name, label in label_map.items():
            folder = DATASET_PATHS["bonn"][class_name]
            signals = load_and_preprocess_bonn(folder, target_fs=SAMPLING_FREQ["bonn"])
            raw_data.extend(signals)
            raw_labels.extend([label] * len(signals))

    elif dataset_name == "hauz":
        for class_name, label in label_map.items():
            folder = DATASET_PATHS["hauz"][class_name]
            signals = load_and_preprocess_hauz(folder, target_fs=SAMPLING_FREQ["hauz"])
            raw_data.extend(signals)
            raw_labels.extend([label] * len(signals))

    data = np.array(raw_data)
    labels = np.array(raw_labels)

    # Shuffle and split
    X_train, X_temp, y_train, y_temp = train_test_split(data, labels, test_size=(val_size + test_size), stratify=labels, random_state=random_state)
    val_ratio = val_size / (val_size + test_size)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=(1 - val_ratio), stratify=y_temp, random_state=random_state)

    return (
        EEGDataset(X_train, y_train, T),
        EEGDataset(X_val, y_val, T),
        EEGDataset(X_test, y_test, T)
    )

def get_dataloaders(task_name, dataset_name, batch_size=32, T=28, **kwargs):
    train_set, val_set, test_set = get_dataset(task_name, dataset_name, T=T, **kwargs)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader

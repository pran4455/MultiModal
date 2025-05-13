import wfdb
import numpy as np
import pandas as pd
import scipy.signal
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns

# Define possible labels
ALL_LABELS = ["H", "HA", "OA", "X", "CA", "CAA", "L", "LA", "A", "MT"]

def load_annotations(record_name):
    annotation = wfdb.rdann(record_name, "st")
    
    def get_labels(ann_str):
        labels = ann_str.split()
        return [label for label in labels if label in ALL_LABELS]
    
    events = [(t, get_labels(a)) for t, a in zip(annotation.sample, annotation.aux_note)]
    
    # Filter out empty labels
    events = [e for e in events if e[1]]
    
    df_events = pd.DataFrame(events, columns=["Time", "Labels"])
    return df_events

def extract_features(record_name):
    record = wfdb.rdrecord(record_name)
    ecg_signal = record.p_signal[:, 0]  # ECG channel
    sampling_rate = record.fs
    
    annotation = wfdb.rdann(record_name, "st")
    rr_intervals = np.diff(annotation.sample) / sampling_rate
    
    lf_hf_ratio = np.var(rr_intervals[:len(rr_intervals)//2]) / (np.var(rr_intervals[len(rr_intervals)//2:]) + 1e-6) if len(rr_intervals) > 1 else 0
    
    f, psd = scipy.signal.welch(ecg_signal, fs=sampling_rate)
    resp_freq = f[np.argmax(psd)] if len(f) > 0 else 0
    
    return {"LF/HF Ratio": lf_hf_ratio, "Respiratory Frequency": resp_freq}

def prepare_data(record_names):
    feature_list = []
    label_list = []
    
    for record_name in record_names:
        df_events = load_annotations(record_name)
        features = extract_features(record_name)
        
        for _, row in df_events.iterrows():
            feature_list.append([features["LF/HF Ratio"], features["Respiratory Frequency"]])
            label_list.append(row["Labels"])  # Keep original labels
    
    return np.array(feature_list), label_list

def train_model(X, y):
    # Convert labels to multi-class format
    y_flat = [label[0] if label else "Unknown" for label in y]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y_flat, test_size=0.2, stratify=y_flat, random_state=42)
    
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"✅ Model Accuracy: {accuracy:.2f}")
    print("\n📊 Classification Report:")
    print(classification_report(y_test, y_pred))
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap="Blues", xticklabels=ALL_LABELS, yticklabels=ALL_LABELS)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()

record_names = [
    "slp01a", "slp01b", "slp02a", "slp02b", "slp03", "slp04",
    "slp14", "slp16", "slp32", "slp37", "slp41", "slp45",
    "slp48", "slp59", "slp60", "slp61", "slp66", "slp67x"
]   # Add more records as needed
X, y = prepare_data(record_names)
train_model(X, y)

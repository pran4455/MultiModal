import wfdb
import numpy as np
import pandas as pd
import scipy.signal
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns

stv = ''

signalfeatures = [
    "rmssd", "sdnn", "nn50", "pnn50", "mrri", "mhr",
    "vlf_power", "lf_power", "hf_power", "lf_hf", "lfnu", "hfnu",
    "edr_vlf_power", "edr_lf_power", "edr_hf_power", "edr_lf_hf", "edr_lfnu", "edr_hfnu",
    "label"
]

# 📌 Step 1: Load Annotations Correctly
LABELS = ["H", "HA", "OA", "X", "CA", "CAA", "L", "LA", "A", "MT"]

def load_annotations(record_name):
    global stv
    annotation = wfdb.rdann(record_name, "st")

    def get_label(c):
        labelcountdict = {label: 0 for label in LABELS}
        global stv
        lst = c.split()
        stv += str(lst) + ' '
        for a in lst:
            if a in labelcountdict:
                labelcountdict[a] += 1
        return labelcountdict

    events = [(t, get_label(a)) for t, a in zip(annotation.sample, annotation.aux_note)]
    df_events = pd.DataFrame(events, columns=["Time", "Class"])
    return df_events

# 📌 Step 2: Extract Features from Multiple Signals
def extract_features(record_name):
    record = wfdb.rdrecord(record_name)
    sampling_rate = record.fs
    
    ecg_signal = record.p_signal[:, 0]  # ECG
    eeg_signal = record.p_signal[:, 1]  # EEG
    eog_signal = record.p_signal[:, 2]  # EOG
    emg_signal = record.p_signal[:, 3]  # EMG
    spo2_signal = record.p_signal[:, 4] if record.p_signal.shape[1] > 4 else np.zeros_like(ecg_signal)

    annotation = wfdb.rdann(record_name, "st")
    rr_intervals = np.diff(annotation.sample) / sampling_rate
    lf_hf_ratio = 0 if len(rr_intervals) < 2 else np.var(rr_intervals[:len(rr_intervals)//2]) / (np.var(rr_intervals[len(rr_intervals)//2:]) + 1e-6)
    
    def compute_frequency_feature(signal):
        f, psd = scipy.signal.welch(signal, fs=sampling_rate)
        return f[np.argmax(psd)] if len(f) > 0 else 0
    
    return {
        "LF/HF Ratio": lf_hf_ratio,
        "Respiratory Frequency": compute_frequency_feature(ecg_signal),
        "EEG Frequency": compute_frequency_feature(eeg_signal),
        "EOG Frequency": compute_frequency_feature(eog_signal),
        "EMG Frequency": compute_frequency_feature(emg_signal),
        "SpO2 Mean": np.mean(spo2_signal)
    }

# 📌 Step 3: Prepare Data
def prepare_data(record_names):
    feature_list = []
    label_list = []

    for record_name in record_names:
        df_events = load_annotations(record_name)
        features = extract_features(record_name)

        for _, row in df_events.iterrows():
            feature_vector = [
                features["LF/HF Ratio"], features["Respiratory Frequency"], 
                features["EEG Frequency"], features["EOG Frequency"], 
                features["EMG Frequency"], features["SpO2 Mean"]
            ]
            label_vector = [row["Class"][label] for label in LABELS]
            feature_list.append(feature_vector)
            label_list.append(label_vector)

    return np.array(feature_list), np.array(label_list)

# 📌 Step 4: Train & Test Multi-Label Classifier
def train_model(X, y):
    if y.shape[1] < 2:
        print("⚠️ Error: Not enough labels for classification.")
        return

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    
    accuracy_per_label = (y_pred == y_test).mean(axis=0)
    print(f"✅ Model Accuracy per Label: {accuracy_per_label}")

    for i, label in enumerate(LABELS):
        print(f"\n📊 Classification Report for {label}:")
        print(classification_report(y_test[:, i], y_pred[:, i]))
        plt.figure(figsize=(4, 4))
        sns.heatmap(confusion_matrix(y_test[:, i], y_pred[:, i]), annot=True, fmt='d', cmap="Blues")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title(f"Confusion Matrix - {label}")
        plt.show()

# 📌 Step 5: Run Full Pipeline
record_names = [
    "slp01a", "slp01b", "slp02a", "slp02b", "slp03", "slp04",
    "slp14", "slp16", "slp32", "slp37", "slp41", "slp45",
    "slp48", "slp59", "slp60", "slp61", "slp66", "slp67x"
]
X, y = prepare_data(record_names)
print("Class Distribution:\n", y.sum(axis=0))
train_model(X, y)

with open('randomv.txt', 'w') as f:
    f.write(stv)


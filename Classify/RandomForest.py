import wfdb
import numpy as np
import pandas as pd
import scipy.signal
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import os
from sklearn.neural_network import MLPClassifier

LABELS = ["H", "HA", "OA", "X", "CA", "CAA", "L", "LA", "A",  "W", "1", "2", "3", "4", "R"]
DATA_DIR = "../data/"

def read_signals(record_name):
    record = wfdb.rdrecord(record_name)
    signals = record.p_signal
    signal_names = record.sig_name
    fs = record.fs
    print(fs)
    return signals, signal_names, fs

def extract_features(signal, fs):
    features = {}
    f, psd = scipy.signal.welch(signal, fs=fs)
    print('f',f,'psd',psd)
    features['mean'] = np.mean(signal)
    features['std'] = np.std(signal)
    features['max'] = np.max(signal)
    features['min'] = np.min(signal)
    features['power'] = np.sum(psd)
    return features

def process_data(record_name, window_size=30):
    signals, signal_names, fs = read_signals(record_name)
    annotation = wfdb.rdann(record_name, 'st')
    segment_length = fs * window_size
    
    data = []
    labels = []
    
    for i in range(0, len(signals), segment_length):
        segment_features = {}
        for j, signal_name in enumerate(signal_names):
            features = extract_features(signals[i:i+segment_length, j], fs)
            segment_features.update({f"{signal_name}_{k}": v for k, v in features.items()})
        
        segment_labels = {label: 0 for label in LABELS}
        for ann_time, ann_note in zip(annotation.sample, annotation.aux_note):
            if i <= ann_time < i + segment_length:
                for label in LABELS:
                    if label in ann_note:
                        segment_labels[label] = 1
        
        data.append(segment_features)
        labels.append(segment_labels)
    
    df = pd.DataFrame(data)
    df_labels = pd.DataFrame(labels)
    df = pd.concat([df, df_labels], axis=1)
    return df

record_names = [
    "slp01a", "slp01b", "slp02a", "slp02b", "slp03", "slp04",
    "slp14", "slp16", "slp32", "slp37", "slp41", "slp45",
    "slp48", "slp59", "slp60", "slp61", "slp66", "slp67x"
]

dfs = []
for record in record_names:
    rec_path = os.path.join(DATA_DIR,record)
    df = process_data(rec_path)
    df.to_csv(f"{record}_features.csv", index=False)
    dfs.append(df)

df_all = pd.concat(dfs, ignore_index=True)

# df_all = df_all.fillna(0)

X = df_all.drop(columns=LABELS)
y = df_all[LABELS]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred, target_names=LABELS))
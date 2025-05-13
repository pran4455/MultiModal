import wfdb
import numpy as np
import pandas as pd
import scipy.signal
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, hamming_loss, multilabel_confusion_matrix
from sklearn.preprocessing import StandardScaler
import os

# Constants - MATCHING ORIGINAL
LABELS = ["H", "HA", "OA", "X", "CA", "CAA", "L", "LA", "A", "W", "1", "2", "3", "4", "R"]
DATA_DIR = "../data/"
WINDOW_SIZE = 30  # seconds

def read_signals(record_name):
    """Read signals and metadata from record"""
    try:
        rec_path = os.path.join(DATA_DIR, record_name)
        record = wfdb.rdrecord(rec_path)
        return record.p_signal, record.sig_name, record.fs
    except Exception as e:
        print(f"Error reading {record_name}: {str(e)}")
        return None, None, None

def extract_features(signal, fs):
    """ORIGINAL FEATURE SET - only basic features"""
    features = {}
    
    # Basic time-domain features
    features['mean'] = np.mean(signal)
    features['std'] = np.std(signal)
    features['max'] = np.max(signal)
    features['min'] = np.min(signal)
    
    # Frequency-domain feature (same as original)
    try:
        f, psd = scipy.signal.welch(signal, fs=fs)
        features['power'] = np.sum(psd)
    except:
        features['power'] = 0
    
    return features

def process_record(record_name, window_size=WINDOW_SIZE):
    """Process a single record into features and labels"""
    signals, signal_names, fs = read_signals(record_name)
    if signals is None:
        return None
    
    segment_length = fs * window_size
    annotation = wfdb.rdann(record_name, 'st')
    
    data = []
    labels = []
    
    for i in range(0, len(signals), segment_length):
        segment = signals[i:i+segment_length]
        if len(segment) == 0:
            continue
            
        segment_features = {}
        for j, sig_name in enumerate(signal_names):
            sig = segment[:, j]
            features = extract_features(sig, fs)
            for stat_name, stat_val in features.items():
                segment_features[f"{sig_name}_{stat_name}"] = stat_val
        
        # Label processing
        segment_labels = {label: 0 for label in LABELS}
        for ann_time, ann_note in zip(annotation.sample, annotation.aux_note):
            if i <= ann_time < i + segment_length:
                for label in LABELS:
                    if label in ann_note:
                        segment_labels[label] = 1
        
        data.append(segment_features)
        labels.append(segment_labels)
    
    return pd.DataFrame(data), pd.DataFrame(labels)

def main():
    record_names = [
        "slp01a", "slp01b", "slp02a", "slp02b", "slp03", "slp04",
        "slp14", "slp16", "slp32", "slp37", "slp41", "slp45",
        "slp48", "slp59", "slp60", "slp61", "slp66", "slp67x"
    ]
    
    # Process all records
    dfs = []
    for record in record_names:
        rec_path = os.path.join(DATA_DIR, record)
        print(f"Processing {record}...")
        df_features, df_labels = process_record(rec_path)
        if df_features is not None:
            df = pd.concat([df_features, df_labels], axis=1)
            df.to_csv(f"{record}_features.csv", index=False)
            dfs.append(df)
    
    # Combine all data
    df_all = pd.concat(dfs, ignore_index=True)
    
    # Minimal NaN handling - only drop rows with all NaN
    print(f"\nOriginal shape: {df_all.shape}")
    df_all = df_all.dropna(how='all')  # Only drop rows where ALL values are NaN
    print(f"After dropping fully NaN rows: {df_all.shape}")
    
    # Fill any remaining NaNs with 0 (simple approach)
    df_all = df_all.fillna(0)
    
    # Prepare features and labels
    X = df_all.drop(columns=LABELS)
    y = df_all[LABELS]
    
    # Feature scaling for MLP
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    
    # Train model
    print("\nTraining MLP classifier...")
    clf = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        max_iter=500,
        random_state=42,
        early_stopping=True,
        verbose=True
    )
    clf.fit(X_train, y_train)
    
    # Evaluate
    y_pred = clf.predict(X_test)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=LABELS))

    accuracies = (y_pred == y_test).mean(axis=0)
    for label, acc in zip(LABELS, accuracies):
        print(f"Accuracy for {label}: {acc:.2f}")

    # 2. Overall accuracy (strict match)
    strict_accuracy = accuracy_score(y_test, y_pred)
    print(f"\nStrict accuracy (all labels must match): {strict_accuracy:.2f}")

if __name__ == "__main__":
    main()
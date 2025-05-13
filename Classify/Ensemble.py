import wfdb
import numpy as np
import pandas as pd
import scipy.signal
import os
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.multioutput import MultiOutputClassifier
from tqdm import tqdm
from xgboost import XGBClassifier

from lightgbm import LGBMClassifier
from sklearn.ensemble import StackingClassifier

LABELS = ["H", "HA", "OA", "X", "CA", "CAA", "L", "LA", "A", "W", "1", "2", "3", "4", "R"]
DATA_DIR = "../data/"

def read_signals(record_name):
    record = wfdb.rdrecord(record_name)
    return record.p_signal, record.sig_name, record.fs

def extract_features(signal, fs):
    f, psd = scipy.signal.welch(signal, fs=fs)
    return {
        'mean': np.mean(signal),
        'std': np.std(signal),
        'max': np.max(signal),
        'min': np.min(signal),
        'power': np.sum(psd)
    }

def process_data(record_name, window_size=30):
    signals, signal_names, fs = read_signals(record_name)
    annotation = wfdb.rdann(record_name, 'st')
    segment_length = int(fs * window_size)

    data, labels = [], []

    for i in range(0, len(signals) - segment_length + 1, segment_length):
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
    return pd.concat([df, df_labels], axis=1)

# Step 1: Load and process data
record_names = [
    "slp01a", "slp01b", "slp02a", "slp02b", "slp03", "slp04",
    "slp14", "slp16", "slp32", "slp37", "slp41", "slp45",
    "slp48", "slp59", "slp60", "slp61", "slp66", "slp67x"
]

dfs = []
print("\n🔍 Extracting features from records...")
for record in tqdm(record_names):
    rec_path = os.path.join(DATA_DIR, record)
    df = process_data(rec_path)
    df.to_csv(f"{record}_features.csv", index=False)
    dfs.append(df)

df_all = pd.concat(dfs, ignore_index=True).fillna(0)

# Step 2: Feature and label split
X = df_all.drop(columns=LABELS)
y = df_all[LABELS]

# Step 3: Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 4: Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Step 5: Define individual models with higher complexity
rf = RandomForestClassifier(n_estimators=500, max_depth=40, random_state=42, n_jobs=-1)
mlp = MLPClassifier(hidden_layer_sizes=(512, 256, 128), activation='relu', solver='adam',
                    max_iter=600, early_stopping=True, random_state=42)
xgb = XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=10,
                    use_label_encoder=False, eval_metric='logloss', verbosity=0, n_jobs=-1)
lgb = LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=64, random_state=42, n_jobs=-1)

# Step 6: Use StackingClassifier instead of simple voting
base_learners = [
    ('rf', rf),
    ('mlp', mlp),
    ('xgb', xgb),
    ('lgb', lgb)
]

meta_learner = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)

# Step 7: Wrap stacked ensemble inside MultiOutputClassifier
stacking_ensemble = MultiOutputClassifier(
    StackingClassifier(estimators=base_learners, final_estimator=meta_learner, n_jobs=-1),
    n_jobs=-1
)

print("\n🧠 Training enhanced stacked model for all labels...")
stacking_ensemble.fit(X_train, y_train)

# Step 8: Evaluate
y_pred = stacking_ensemble.predict(X_test)
print("\n📊 Classification Report:\n")
print(classification_report(y_test, y_pred, target_names=LABELS))
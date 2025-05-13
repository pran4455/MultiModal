import wfdb
import numpy as np
import pandas as pd
import scipy.signal
import cupy as cp  # GPU-accelerated NumPy
import cudf  # GPU-accelerated pandas
from cuml.ensemble import RandomForestClassifier as cuRF  # GPU Random Forest
from cuml.linear_model import LogisticRegression as cuLogisticRegression
from cuml.neighbors import KNeighborsClassifier as cuKNN
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import os
from numba import cuda  # For custom GPU functions

# Check GPU availability
def check_gpu():
    try:
        import cupy
        print("CuPy is available - GPU acceleration enabled")
        return True
    except ImportError:
        print("CuPy not available - falling back to CPU")
        return False

HAS_GPU = check_gpu()

LABELS = ["H", "HA", "OA", "X", "CA", "CAA", "L", "LA", "A", "W", "1", "2", "3", "4", "R"]
DATA_DIR = "../data/"

def read_signals(record_name):
    record = wfdb.rdrecord(record_name)
    signals = record.p_signal
    signal_names = record.sig_name
    fs = record.fs
    return signals, signal_names, fs

@cuda.jit
def gpu_sample_entropy(signal, result, m=2, r=0.2):
    """GPU-accelerated sample entropy calculation"""
    i = cuda.grid(1)
    N = signal.shape[0]
    
    if i >= N - m:
        return
    
    std = 0.0
    mean = 0.0
    
    # Compute mean and std in one pass
    for j in range(N):
        mean += signal[j]
    mean /= N
    
    for j in range(N):
        std += (signal[j] - mean) ** 2
    std = (std / N) ** 0.5
    
    if std == 0:
        result[i] = 0
        return
    
    r_val = r * std
    
    # Compute phi(m)
    B = 0.0
    for j in range(N - m + 1):
        max_dist = 0.0
        for k in range(m):
            max_dist = max(max_dist, abs(signal[i + k] - signal[j + k]))
        if max_dist <= r_val:
            B += 1
    
    # Compute phi(m+1)
    A = 0.0
    for j in range(N - m):
        max_dist = 0.0
        for k in range(m + 1):
            max_dist = max(max_dist, abs(signal[i + k] - signal[j + k]))
        if max_dist <= r_val:
            A += 1
    
    if B > 0 and A > 0:
        result[i] = -cp.log(A / B)
    else:
        result[i] = 0

def extract_features(signal, fs):
    features = {}
    
    if HAS_GPU:
        signal_gpu = cp.asarray(signal)
    else:
        signal_gpu = np.asarray(signal)
    
    # Time-domain features
    features['mean'] = float(cp.mean(signal_gpu)) if HAS_GPU else np.mean(signal_gpu)
    features['std'] = float(cp.std(signal_gpu)) if HAS_GPU else np.std(signal_gpu)
    features['skew'] = float(cp.stats.skew(signal_gpu)) if HAS_GPU else scipy.stats.skew(signal)
    features['kurtosis'] = float(cp.stats.kurtosis(signal_gpu)) if HAS_GPU else scipy.stats.kurtosis(signal)
    features['max'] = float(cp.max(signal_gpu)) if HAS_GPU else np.max(signal)
    features['min'] = float(cp.min(signal_gpu)) if HAS_GPU else np.min(signal)
    features['ptp'] = float(cp.ptp(signal_gpu)) if HAS_GPU else np.ptp(signal)
    features['rms'] = float(cp.sqrt(cp.mean(signal_gpu**2))) if HAS_GPU else np.sqrt(np.mean(signal**2))
    
    # Frequency-domain features (still on CPU as cuSignal is experimental)
    f, psd = scipy.signal.welch(signal, fs=fs)
    features['power'] = np.sum(psd)
    features['dominant_freq'] = f[np.argmax(psd)]
    features['spectral_entropy'] = scipy.stats.entropy(psd)
    
    # Non-linear features with GPU acceleration
    if HAS_GPU:
        result = cp.zeros(1)
        gpu_sample_entropy[1, 1](signal_gpu, result)
        features['sample_entropy'] = float(result[0])
    else:
        features['sample_entropy'] = compute_sample_entropy(signal)
    
    return features

def compute_sample_entropy(signal, m=2, r=0.2):
    """Fallback CPU implementation"""
    N = len(signal)
    std = np.std(signal)
    if std == 0:
        return 0
    r *= std
    
    def _maxdist(x_i, x_j):
        return max([abs(ua - va) for ua, va in zip(x_i, x_j)])
    
    def _phi(m):
        x = [[signal[j] for j in range(i, i + m)] for i in range(N - m + 1)]
        C = [len([1 for x_j in x if _maxdist(x_i, x_j) <= r]) for x_i in x]
        return sum(C) / (N - m + 1.0)
    
    if N <= m:
        return 0
    return -np.log(_phi(m + 1) / _phi(m))

def process_data(record_name, window_size=30):
    signals, signal_names, fs = read_signals(record_name)
    annotation = wfdb.rdann(record_name, 'st')
    segment_length = int(fs * window_size)
    
    data = []
    labels = []
    
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
    
    if HAS_GPU:
        df = cudf.DataFrame(data)
        df_labels = cudf.DataFrame(labels)
        df = cudf.concat([df, df_labels], axis=1)
    else:
        df = pd.DataFrame(data)
        df_labels = pd.DataFrame(labels)
        df = pd.concat([df, df_labels], axis=1)
    
    return df

# Main execution
if __name__ == "__main__":
    record_names = [
        "slp01a", "slp01b", "slp02a", "slp02b", "slp03", "slp04",
        "slp14", "slp16", "slp32", "slp37", "slp41", "slp45",
        "slp48", "slp59", "slp60", "slp61", "slp66", "slp67x"
    ]

    dfs = []
    for record in record_names:
        rec_path = os.path.join(DATA_DIR, record)
        df = process_data(rec_path)
        df.to_csv(f"{record}_features.csv", index=False)
        dfs.append(df)

    if HAS_GPU:
        df_all = cudf.concat(dfs, ignore_index=True)
        df_all = df_all.fillna(0)
        X = df_all.drop(columns=LABELS).to_pandas().values  # Convert to numpy for sklearn
        y = df_all[LABELS].to_pandas().values
    else:
        df_all = pd.concat(dfs, ignore_index=True)
        df_all = df_all.fillna(0)
        X = df_all.drop(columns=LABELS).values
        y = df_all[LABELS].values

    # Feature scaling (on GPU if available)
    if HAS_GPU:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    # Define GPU-accelerated models where available
    models = {
        'XGBoost-GPU': XGBClassifier(
            tree_method='gpu_hist',  # Use GPU acceleration
            predictor='gpu_predict',
            n_estimators=500,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42
        ),
        'LightGBM-GPU': LGBMClassifier(
            device='gpu',  # Use GPU acceleration
            n_estimators=500,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42
        ),
        'Random Forest-GPU': cuRF(
            n_estimators=500,
            max_depth=8,
            random_state=42
        ) if HAS_GPU else RandomForestClassifier(
            n_estimators=500,
            max_depth=8,
            random_state=42,
            n_jobs=-1
        ),
        'Logistic Regression-GPU': cuLogisticRegression(
            penalty='l2',
            C=1.0,
            max_iter=1000,
            random_state=42
        ) if HAS_GPU else LogisticRegression(
            penalty='l2',
            C=1.0,
            max_iter=1000,
            random_state=42,
            n_jobs=-1
        )
    }

    # Train and evaluate each model
    for name, model in models.items():
        print(f"\n{'='*40}\nTraining {name}\n{'='*40}")
        
        # Multi-output classification
        multi_model = MultiOutputClassifier(model, n_jobs=-1 if not HAS_GPU else None)
        multi_model.fit(X_train, y_train)
        
        y_pred = multi_model.predict(X_test)
        
        print(f"\nClassification Report for {name}:\n")
        print(classification_report(y_test, y_pred, target_names=LABELS, zero_division=0))
import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

# Constants
DATA_DIR = "./csv_data"  # Update this to the correct path where CSVs are stored
LABELS = ["H", "HA", "OA", "X", "CA", "CAA", "L", "LA", "A", "W", "1", "2", "3", "4", "R"]

# Streamlit Config
st.set_page_config(
    page_title="Polysomnography Classification",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Sidebar for File Selection
st.sidebar.title("Select a Polysomnography File")
csv_files = [f for f in os.listdir(DATA_DIR) if f.endswith("_features.csv")]
selected_file = st.sidebar.selectbox("Choose a CSV file", csv_files)

if selected_file:
    st.title(f"Analysis for {selected_file}")
    df = pd.read_csv(os.path.join(DATA_DIR, selected_file))
    
    # Splitting Data
    X = df.drop(columns=LABELS)
    y = df[LABELS]
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = joblib.load('StackingModel.pkl')
    scaler = joblib.load('ScalerEnsemble.pkl')
    feature_names = joblib.load('FeatureEnsemble.pkl')

    # Train MLP Classifier
    #clf = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500, random_state=42)
    #clf.fit(X_train, y_train)
    X = df.drop(columns=LABELS)
    X = X.reindex(columns=feature_names, fill_value=0)

    # Scale features using training scaler
    X_scaled = scaler.transform(X)

    # Predict
    y_pred = model.predict(X_scaled)
    
    # Display Classification Report
    st.subheader("Classification Report")
    report = classification_report(y, y_pred, target_names=LABELS, output_dict=True)
    st.dataframe(pd.DataFrame(report).transpose())
    
    # Data Overview
    st.subheader("Feature Data Overview")
    st.write(df.head())
    
    # Feature Distribution Visualization
    st.subheader("Feature Distributions")
    selected_feature = st.selectbox("Choose a Feature to Visualize", X.columns)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(df[selected_feature], bins=30, color='blue', edgecolor='black', alpha=0.7)
    ax.set_title(f"Distribution of {selected_feature}")
    ax.set_xlabel("Value")
    ax.set_ylabel("Frequency")
    st.pyplot(fig)
    
    valid_features = X.columns[X.astype(bool).sum(axis=0) > 5]  # Use `X.count()` if you want non-NaN instead

    # Subset the feature matrix
    X_filtered = X[valid_features]

    # Compute correlation matrix on filtered features
    feature_corr = X_filtered.corr()

    # Plot
    fig, ax = plt.subplots(figsize=(12, 8))
    cax = ax.matshow(feature_corr, cmap='coolwarm')
    plt.xticks(range(len(feature_corr.columns)), feature_corr.columns, rotation=90)
    plt.yticks(range(len(feature_corr.columns)), feature_corr.columns)
    plt.colorbar(cax)
    st.pyplot(fig)
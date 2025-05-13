import os
import warnings
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import wfdb
from biosppy.signals.ecg import correct_rpeaks, hamilton_segmenter
from biosppy.signals.tools import filter_signal
from hrv.classical import frequency_domain, time_domain
from scipy.signal import medfilt
import pandas as pd
import joblib
warnings.filterwarnings("ignore")

# Constants
base_dir = r"../data/"
fs = 100
hr_min = 20
hr_max = 300

# Streamlit Config
st.set_page_config(
    page_title="ECG Analysis Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Sidebar
st.sidebar.title("ECG Analysis Menu")
recording_input = st.sidebar.text_input("Enter Recording Name (e.g., 'slp01a')", "")

st.sidebar.markdown("---")
st.sidebar.markdown("#### Analysis Options")
show_filtered_signal = st.sidebar.checkbox("Show Raw vs Filtered Signal", value=True)
show_rpeaks = st.sidebar.checkbox("Show R-Peak Detection", value=True)
show_histogram = st.sidebar.checkbox("Show R-R Interval Histogram", value=True)
show_hrv_time_features = st.sidebar.checkbox("Show Time-Domain HRV Metrics", value=True)
show_hrv_frequency_features = st.sidebar.checkbox("Show Frequency-Domain HRV Metrics", value=True)

# Main Title and Description
st.title("ECG R-Peak Detection and HRV Analysis Dashboard")
st.markdown(""" 
    Analyze ECG signals with advanced visualizations. Perform R-Peak detection, view HRV metrics, 
    and explore detailed frequency and time-domain features.
""")

# Visualization functions
def plot_signal(raw_signal, filtered_signal):
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(raw_signal, label="Raw Signal", alpha=0.6)
    ax.plot(filtered_signal, label="Filtered Signal", alpha=0.8)
    ax.set_title("ECG Signal (Raw vs Filtered)")
    ax.set_xlabel("Sample")
    ax.set_ylabel("Amplitude")
    ax.legend()
    st.pyplot(fig)

def plot_rpeaks(segment, rpeaks):
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(segment, label="Filtered ECG", color="dodgerblue")
    ax.scatter(rpeaks, segment[rpeaks], color="red", label="R-peaks", marker="o")
    ax.set_title("R-Peak Detection")
    ax.set_xlabel("Sample")
    ax.set_ylabel("Amplitude")
    ax.legend()
    st.pyplot(fig)

def plot_histogram(rri):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(rri * 1000, bins=30, color="purple", edgecolor="black")
    ax.set_title("R-R Interval Histogram")
    ax.set_xlabel("R-R Interval (ms)")
    ax.set_ylabel("Frequency")
    st.pyplot(fig)

def plot_time_domain_metrics(time_features):
    st.subheader("Time-Domain HRV Metrics")
    
    # Prepare data for visualization
    metrics = {
        'Mean RR (ms)': time_features['mrri'],
        'Mean HR (bpm)': time_features['mhr'],
        'SDNN (ms)': time_features['sdnn'],
        'RMSSD (ms)': time_features['rmssd'],
        'NN50': time_features['nn50'],
        'pNN50 (%)': time_features['pnn50']
    }
    
    # Create two columns for display
    col1, col2 = st.columns(2)
    
    with col1:
        # Display metrics as a table
        st.table(pd.DataFrame.from_dict(metrics, orient='index', columns=['Value']))
    
    with col2:
        # Create a bar chart
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(metrics.keys(), metrics.values(), color='skyblue')
        ax.set_title("Time-Domain HRV Metrics")
        ax.set_ylabel("Value")
        plt.xticks(rotation=45)
        st.pyplot(fig)

def plot_frequency_domain_metrics(freq_features):
    st.subheader("Frequency-Domain HRV Metrics")
    
    # Prepare data for visualization
    metrics = {
        'Total Power (ms²)': freq_features['total_power'],
        'VLF Power (ms²)': freq_features['vlf'],
        'LF Power (ms²)': freq_features['lf'],
        'HF Power (ms²)': freq_features['hf'],
        'LF/HF Ratio': freq_features['lf_hf'],
        'LFnu': freq_features['lfnu'],
        'HFnu': freq_features['hfnu']
    }
    
    # Create two columns for display
    col1, col2 = st.columns(2)
    
    with col1:
        # Display metrics as a table
        st.table(pd.DataFrame.from_dict(metrics, orient='index', columns=['Value']))
    
    with col2:
        # Create a bar chart for spectral components
        fig, ax = plt.subplots(figsize=(8, 4))
        components = ['VLF', 'LF', 'HF']
        values = [freq_features['vlf'], freq_features['lf'], freq_features['hf']]
        ax.bar(components, values, color=['lightblue', 'mediumseagreen', 'salmon'])
        ax.set_title("Spectral Power Distribution")
        ax.set_ylabel("Power (ms²)")
        st.pyplot(fig)

def process_hrv_analysis(segment, fs):
    """Process ECG segment and return HRV metrics"""
    # Filter the signal
    filtered, _, _ = filter_signal(segment, ftype='FIR', band='bandpass', 
                                 order=int(0.3 * fs), frequency=[3, 45], sampling_rate=fs)
    
    # Detect R-peaks
    rpeaks, = hamilton_segmenter(filtered, sampling_rate=fs)
    rpeaks, = correct_rpeaks(filtered, rpeaks, sampling_rate=fs, tol=0.1)
    
    # Calculate RR intervals
    rri = np.diff(rpeaks) / fs  # Convert to seconds
    rri_tm = rpeaks[1:] / fs  # Time points
    
    # Clean RR intervals
    rri = medfilt(rri, kernel_size=3)
    
    # Calculate HRV metrics
    time_features = time_domain(rri * 1000)  # Convert to milliseconds
    freq_features = frequency_domain(rri, rri_tm)
    
    return time_features, freq_features, rpeaks, rri

# Streamlit app functionality
if recording_input:
    recording = recording_input.strip()
    try:
        st.header(f"Analysis for recording: {recording}")
        
        # Read signal
        record = wfdb.rdrecord(os.path.join(base_dir, recording), channels=[0])
        signal = record.p_signal[:, 0]
        
        # Process first 5 minutes for demonstration
        segment = signal[:fs*60]  # 5 minutes
        
        # Perform HRV analysis
        time_features, freq_features, rpeaks, rri = process_hrv_analysis(segment, fs)
        
        # Show visualizations based on user selection
        if show_filtered_signal:
            raw_segment = segment.copy()
            filtered, _, _ = filter_signal(segment, ftype='FIR', band='bandpass', 
                                        order=int(0.3 * fs), frequency=[3, 45], sampling_rate=fs)
            plot_signal(raw_segment, filtered)
        
        if show_rpeaks:
            filtered, _, _ = filter_signal(segment, ftype='FIR', band='bandpass', 
                                        order=int(0.3 * fs), frequency=[3, 45], sampling_rate=fs)
            plot_rpeaks(filtered, rpeaks)
        
        if show_histogram:
            plot_histogram(rri)
        
        if show_hrv_time_features:
            plot_time_domain_metrics(time_features)
        
        if show_hrv_frequency_features:
            plot_frequency_domain_metrics(freq_features)
        
        st.success(f"Analysis completed for {recording}")
        
    except Exception as e:
        st.error(f"Error processing recording: {str(e)}")
else:
    st.info("Please enter a recording name in the sidebar to begin analysis")
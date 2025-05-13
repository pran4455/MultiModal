# MultiModal Sleep Apnea Detection System

A comprehensive sleep apnea detection and analysis system that leverages multiple physiological signals including ECG, EEG, EOG, EMG, and SpO₂ for accurate diagnosis and monitoring.

## 🌟 Features

### Signal Processing & Analysis
- **Multi-Signal Integration**: Processes ECG, EEG, EOG, EMG, and SpO₂ signals simultaneously
- **R-Peak Detection**: Advanced ECG R-peak detection with noise filtering
- **HRV Analysis**: Comprehensive Heart Rate Variability analysis in both time and frequency domains
- **Real-time Processing**: Capable of processing signals in real-time for immediate feedback

### Machine Learning Models
- **Ensemble Classification**: Utilizes a stacked ensemble of:
  - Random Forest
  - Neural Networks (MLP)
  - XGBoost
  - LightGBM
- **Multi-Label Classification**: Detects multiple sleep stages and events:
  - Sleep Stages (W, 1, 2, 3, 4, R)
  - Apnea Events (H, HA, OA, CA, CAA)
  - Other Events (L, LA, A, X)

### Interactive Dashboard
- **Signal Visualization**: Real-time display of all physiological signals
- **Feature Extraction**: Automated extraction of relevant biomarkers
- **Statistical Analysis**: Comprehensive statistical insights and trends
- **Report Generation**: Detailed analysis reports for clinical use

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- Required packages can be installed via:
```bash
pip install -r requirements.txt
```

### Model File Generation
Before running the web dashboard, you need to generate the required model files:

1. Navigate to the `Classify` directory:
```bash
cd Classify
```

2. Run the ensemble training notebook or Python script:
   Using the notebook: `ensemble copy.ipynb`

This will generate the following required `.pkl` files:
- `StackingModel.pkl` (or `stacking_model.pkl`): The trained stacking ensemble model
- `ScalerEnsemble.pkl` (or `scaler.pkl`): The fitted StandardScaler for feature normalization
- `FeatureEnsemble.pkl` (or `feature_columns.pkl`): The list of feature columns

These files are required for the Streamlit dashboard to function properly. Make sure they are generated before starting the web application.

### Directory Structure
```
MultiModal/
├── Web/                 # Web dashboard components
│   ├── pages/          # Dashboard pages
│   └── assets/         # Static assets
├── Classify/           # Classification models
│   ├── Ensemble.py     # Ensemble model implementation
│   └── mlp.py         # Neural network implementation
├── data/               # Dataset directory
└── assets/             # Project assets
```

### Running the Application
1. Start the web dashboard:
```bash
cd Web
streamlit run "Welcome Dashboard.py"
```

2. Access the dashboard at `http://localhost:8501`

## 📊 Features & Metrics

The system extracts various features from physiological signals:

### Time Domain Features
- Mean, Standard Deviation
- Maximum and Minimum values
- Root Mean Square (RMS)
- Peak-to-Peak Amplitude

### Frequency Domain Features
- Power Spectral Density
- Dominant Frequency
- Spectral Entropy
- VLF, LF, HF Power bands
- LF/HF Ratio

### Non-linear Features
- Sample Entropy
- Spectral Features
- Signal Complexity Measures

## 🎯 Performance

The system utilizes a stacked ensemble approach combining multiple classifiers:
- Random Forest (500 estimators)
- Neural Network (512-256-128 architecture)
- XGBoost (300 estimators)
- LightGBM (300 estimators)

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Contact

For questions and feedback, please open an issue in the repository.

## 🙏 Acknowledgments

- Sleep-EDF Database contributors
- PhysioNet for providing sleep study datasets
- Open-source community for various tools and libraries used in this project

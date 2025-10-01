# Grid Fault Detection System

Detect partial discharge patterns in power line signals using ML classifiers. Reduce maintenance costs and prevent outages through automated monitoring and AI-powered diagnostics reporting.

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Kaggle](https://img.shields.io/badge/dataset-VSB%20Power%20Line-orange.svg)](https://www.kaggle.com/competitions/vsb-power-line-fault-detection)

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Installation](#installation)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Overview

This project implements an end-to-end machine learning pipeline for detecting partial discharge faults in medium voltage power transmission systems using the VSB Power Line Fault Detection dataset from Kaggle. 

### Problem Statement

Partial discharge is a localized dielectric breakdown in power lines that doesn't completely bridge conductors but progressively damages insulation, eventually leading to:
- Complete equipment failure
- Widespread power outages
- Fire hazards
- Millions of dollars in repair costs

Manual inspection across hundreds of miles of power lines is impractical, costly, and reactive rather than proactive.

### Our Solution

An automated system that:
- Analyzes 800,000-point electrical signal measurements
- Detects fault patterns with >40% Matthews Correlation Coefficient (MCC) and > 0.5 PR-AUC
- Provides real-time analysis with <10ms inference latency
- Generates professional diagnostic reports with AI-generated insights using GenAI
- Enables proactive maintenance scheduling
- Reduces operational costs through early detection

  ![ML STAGES ](https://github.com/fcUalberta/grid_fault_detection/blob/main/images/ML%20stages.png)

## Key Features

### Machine Learning Pipeline
- **Multi-Model Framework**: XGBoost,  CatBoost, Random Forest, Logistic Regression
- **Tier-Based Performance**: Models optimized from ultra-fast (<1ms) to high-accuracy (<50ms)
- **Imbalanced Data Handling**: Advanced techniques for rare fault detection (6% positive class)
- **Comprehensive Evaluation**: MCC, PR-AUC, Latency optimized for power systems

### Key Visuals from Exploratory Data Analysis

 ![Normal vs Fault Signal ](https://github.com/fcUalberta/grid_fault_detection/blob/main/images/Normal%20vs%20Fault%20Visual.png)
 ![Time Series Pattern Analysis using Sliding Windows](https://github.com/fcUalberta/grid_fault_detection/blob/main/images/timeseries%20pattern.png)
  
### Signal Processing & Feature Engineering
- **50+ Domain-Specific Features**: Time domain, frequency domain, and wavelet transforms
- **Power System Metrics**: RTHD, Transient detection, amplitude variation etc.
- **Advanced Preprocessing**: Spectral Centroid, wavelet decomposition, autocorrelation, dominant frequency analysis
- **Dimensionality Reduction**: Chunking strategies for 800,000-point signals

###  Model Performance Comparison
![Model Performance Comparison Across Metrics](https://github.com/fcUalberta/grid_fault_detection/blob/main/images/model_eval.png)
![Latency Vs Accuracy Trade-Off for Grid](https://github.com/fcUalberta/grid_fault_detection/blob/main/images/models%20for%20grid.png)

### Explainable AI
- **SHAP Integration**: Feature importance and prediction interpretation
- **Model Transparency**: Understanding which signal characteristics indicate faults
- **Operational Insights**: Actionable explanations for grid operators

![Single Instance Prediction Explanation](https://github.com/fcUalberta/grid_fault_detection/blob/main/images/SHAP.png)

### GenAI-Powered Diagnostics
- **Automated PDF Reports**: Professional documentation for stakeholders
- **Natural Language Insights**: AI-generated analysis using Hugging Face models
- **Trend Analysis**: Historical comparisons and pattern detection
- **Alert System**: Intelligent fault prioritization and maintenance recommendations

## Installation

### Prerequisites

- Python 3.8 or higher
- 8GB RAM minimum (16GB recommended for full dataset)
- Optional: CUDA-capable GPU for deep learning models

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/fcUalberta/grid_fault_detection.git
cd grid_fault_detection
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

### Core Dependencies

```
# Data Processing
pandas>=1.3.0
numpy>=1.20.0
scipy>=1.7.0

# Machine Learning
scikit-learn>=1.0.0
lightgbm>=3.3.0
xgboost>=1.5.0
catboost>=1.0.0

# Signal Processing
pywavelets>=1.1.0

# Visualization & Reporting
matplotlib>=3.3.0
seaborn>=0.11.0
reportlab>=3.6.0

# Explainability
shap>=0.40.0

# GenAI (Optional)
transformers>=4.20.0
torch>=1.11.0

# Model Persistence
joblib>=1.1.0
```


## Dataset

### Source

The VSB Power Line Fault Detection dataset from Kaggle Competition:
[https://www.kaggle.com/competitions/vsb-power-line-fault-detection](https://www.kaggle.com/competitions/vsb-power-line-fault-detection)

### Download Instructions

1. **Install Kaggle API**:
```bash
pip install kaggle
```

2. **Set up Kaggle credentials**:
   - Go to Kaggle Account Settings
   - Create new API token (downloads kaggle.json)
   - Place in `~/.kaggle/kaggle.json` (Linux/Mac) or `C:\Users\<username>\.kaggle\kaggle.json` (Windows)

3. **Download dataset**:
```bash
kaggle competitions download -c vsb-power-line-fault-detection
unzip vsb-power-line-fault-detection.zip -d data/
```

### Data Structure

```
data/
├── train.parquet          # 8,712 signals × 800,000 measurements each
├── test.parquet           # 20,337 signals for prediction
├── metadata_train.csv     # Signal IDs, phase IDs, target labels
└── metadata_test.csv      # Test metadata
```
Link to the train and test parquet files are provided in the text file due to the size

### Data Characteristics

- **Signals**: 8,712 training signals, 20,337 test signals
- **Measurements per Signal**: 800,000 floating-point values
- **Target**: Binary (0 = normal operation, 1 = partial discharge detected)
- **Class Imbalance**: ~94% normal, ~6% faults (typical for power systems)
- **Phases**: 3-phase electrical system representation

## Project Structure

```
grid_fault_detection/
│
├── data/                          # Dataset directory (download separately)
│   ├── train.parquet (link to download this file is provided)
│   ├── metadata_train.csv
│   └── metadata_test.csv
│
├── helper/                      # Source code modules
│   ├── common.py                # common modules across ML stages
│   ├── evaluations.py           # model evaluations and visualizations
│   ├── load_save_model.py       # loading and saving models
│   ├── report_generation.py     #PDF report generation using SLM
│   ├── shap_explain.py          # model explanations using SHAP
│
├── notebooks/                     # Jupyter notebooks for analysis
│   ├── 01_exploratory_data_analysis.ipynb (Since the file is really big, uploaded it as an HTML for preview)
│   ├── 02_signal_visualization.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_model_training_comparison.ipynb
│   └── 05_explainability_analysis.ipynb
│
├── models/                        # Saved trained models
│   └── (generated during training)
│
│── saved_models/                  # Saved best trained models after comparison
│   └── (generated during training)
│
├── diagnostic_reports/             # Generated diagnostic reports
│   ├── diagnostics_*.pdf
│   ├── analysis_data_*.json
│   └── visualizations/
│
│
├── requirements.txt               # Python dependencies
├── README.md                      # This file
├── LICENSE                        # MIT License
└── .gitignore                     # Git ignore rules
```


## GenAI Integration

The system uses **Hugging Face transformer models** for intelligent insight generation:

```python
# Example AI-generated insight:
"""
⚠️ ELEVATED ALERT: Fault rate of 7.3% exceeds normal operational 
threshold (5%). Analysis identifies 23 signals showing fault patterns 
with average confidence of 0.82. Peak fault activity observed during 
hours 14-16, suggesting load-related stress. Immediate inspection 
recommended for Signal IDs: 3456, 7821, 2901. Historical comparison 
shows 45% increase vs. last period, indicating potential systematic 
issue requiring root cause analysis.
"""
```

## License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **VSB Technical University of Ostrava** for providing the power line fault detection dataset
- **Kaggle** for hosting the competition and facilitating data science collaboration
- **Open Source Community** for excellent libraries:
  - scikit-learn, XGBoost, LightGBM, CatBoost for ML models
  - SHAP for model interpretability
  - Hugging Face for transformer models and GenAI capabilities
  - ReportLab for professional PDF generation
  - Matplotlib and Seaborn for visualization


---

*

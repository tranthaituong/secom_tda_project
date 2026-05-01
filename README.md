# SECOM TDA Project
## Anomaly Detection in Semiconductor Manufacturing using Topological Data Analysis

---

## Table of Contents

1. [Overview](#overview)
2. [Project Structure](#project-structure)
3. [Installation](#installation)
4. [Quick Start](#quick-start)
5. [Pipeline Steps](#pipeline-steps)
6. [Parameters](#parameters)
7. [Output Files](#output-files)
8. [Examples](#examples)
9. [Understanding the Results](#understanding-the-results)

---

## Overview

This project analyzes the **SECOM (Semiconductor Manufacturing) dataset** using **Topological Data Analysis (TDA)** for anomaly detection in manufacturing processes.

### Dataset

- **SECOM Dataset**: 1567 samples × 591 features
- **Anomaly Rate**: ~6.6% (104 failures out of 1567)
- **Goal**: Detect anomalies (equipment failures) in semiconductor manufacturing

### Methodology

1. **Data Processing**: Handle missing values, remove constant features, standardize, apply PCA
2. **Time-Series Windows**: Create sliding windows to capture temporal patterns
3. **TDA Features**: Extract H1 (loops/cycles) features using Persistent Homology
4. **ML Baselines**: Compare with Isolation Forest and One-Class SVM
5. **Evaluation**: Compare performance using F1-Score, AUC, Precision, Recall

---

## Project Structure

```
secom_tda_project/
├── main.py                 # Main entry point
├── requirements.txt         # Python dependencies
├── README.md              # This file
│
├── modules/               # Python modules
│   ├── __init__.py       # Module exports
│   ├── config.py         # Configuration and default parameters
│   ├── data_processing.py # NV1: Data loading and preprocessing
│   ├── tda_features.py   # NV3: TDA feature extraction
│   ├── ml_baselines.py   # NV4: ML baseline models
│   ├── evaluation.py     # NV5: Evaluation metrics
│   └── visualization.py  # NV6: Plots and reports
│
├── data/                  # Input data (put SECOM files here)
│   ├── secom.data
│   └── secom_labels.data
│
└── outputs/               # Generated outputs
    ├── data_pca.npy
    ├── windows_dict.npy
    ├── topo_features.npy
    ├── topo_scores.npy
    ├── ml_preds.npy
    ├── ablation_results.csv
    ├── topo_ablation_heatmap.png
    ├── metric_comparison.png
    └── analysis_notes.md
```

---

## Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### Step 1: Clone or Copy the Project

```bash
# If using git
git clone <repository-url>
cd secom_tda_project

# Or copy the folder directly
```

### Step 2: Create Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate on Windows
venv\Scripts\activate

# Activate on Linux/Mac
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Prepare Data

Copy your SECOM data files to the `data/` folder:

```
data/
├── secom.data         # Main data (1567 × 591)
└── secom_labels.data  # Labels (-1=normal, 1=anomaly)
```

If your data is located elsewhere, you can specify the path using command-line arguments.

---

## Quick Start

### Run Full Pipeline

```bash
python main.py --all
```

This will:
1. Process SECOM data
2. Extract TDA features
3. Run ML baselines
4. Evaluate and compare models
5. Generate visualizations

### Run Step by Step

```bash
# Step 1: Data processing
python main.py --data

# Step 2: TDA features
python main.py --tda

# Step 3: ML baselines
python main.py --ml

# Step 4: Evaluation
python main.py --eval

# Step 5: Visualization
python main.py --viz
```

---

## Pipeline Steps

### Step 1: Data Processing (NV1)

Loads raw SECOM data and performs preprocessing:

1. **Load Data**: Read `secom.data` and `secom_labels.data`
2. **Handle Missing Values**: Impute NaN with column means
3. **Remove Constant Features**: Remove features with zero variance
4. **Standardize**: Apply StandardScaler (z-score normalization)
5. **PCA**: Reduce dimensions to [2, 3, 5] components
6. **Sliding Windows**: Create time-series windows with sizes [20, 30, 50]

**Output**: `windows_dict.npy` containing preprocessed data

### Step 2: TDA Features (NV3)

Extracts topological features using Persistent Homology:

1. **Rips Filtration**: Build simplicial complexes at multiple scales
2. **Persistent Homology**: Track H1 (loop) features
3. **Persistence Scores**: Calculate persistence = death - birth
4. **Anomaly Threshold**: Use 95th percentile as threshold

**Output**: `topo_features.npy`, `topo_scores.npy`, `topo_diagrams.npy`

### Step 3: ML Baselines (NV4)

Runs traditional ML anomaly detection models:

1. **Isolation Forest**: With contamination = [0.01, 0.05, 0.07, 0.1]
2. **One-Class SVM**: With nu = [0.01, 0.05, 0.07, 0.1]

**Output**: `ml_preds.npy`, `ml_param_log.csv`

### Step 4: Evaluation (NV5)

Evaluates all models using ablation study:

- **Metrics**: Precision, Recall, F1-Score, AUC
- **Configurations**: All combinations of PCA × Window Size × Parameters
- **Comparison**: Topology vs. Isolation Forest vs. One-Class SVM

**Output**: `ablation_results.csv`

### Step 5: Visualization (NV6)

Generates visualizations and reports:

1. **Ablation Heatmap**: F1-Score by PCA and Window Size
2. **Metric Comparison**: Bar chart comparing models
3. **Persistence Diagrams**: Side-by-side comparison
4. **Analysis Notes**: Markdown summary

**Output**: PNG files, PDF report, markdown notes

---

## Parameters

### Data Processing Parameters

| Parameter | Flag | Default | Description |
|-----------|------|---------|-------------|
| Data file | `--data-file` | `data/secom.data` | Path to SECOM data |
| Labels file | `--labels-file` | `data/secom_labels.data` | Path to labels |
| PCA components | `--pca` | `[2, 3, 5]` | Number of PCA components |
| Window sizes | `--windows` | `[20, 30, 50]` | Sliding window lengths |
| Output directory | `--output-dir` | `outputs/` | Output folder |

### TDA Parameters

| Parameter | Flag | Default | Description |
|-----------|------|---------|-------------|
| Threshold percentile | `--percentile` | `95.0` | Percentile for anomaly threshold |

### Other Parameters

| Parameter | Flag | Default | Description |
|-----------|------|---------|-------------|
| Log file | `--log-file` | (console only) | Path to log file |
| Verbose | `--verbose` | False | Enable debug logging |

### Example with Custom Parameters

```bash
# Use custom PCA and window sizes
python main.py --data --pca 2 4 6 --windows 10 20 30

# Use different anomaly threshold
python main.py --tda --percentile 90

# Save logs to file
python main.py --all --log-file outputs/pipeline.log
```

---

## Default Parameter Values

### PCA Components
- **2**: Aggressive dimensionality reduction, captures main variation
- **3**: Slightly more detail
- **5**: More features, potentially more noise

### Window Sizes (L)
- **20**: Short-term patterns, faster response to changes
- **30**: Balanced window (recommended as optimal)
- **50**: Long-term patterns, smoother but may miss quick anomalies

### Contamination Fractions
- **0.01**: Very few anomalies expected (1%)
- **0.05**: Moderate (5%)
- **0.07**: Standard (7%)
- **0.1**: Higher rate expected (10%)

### Threshold Percentile
- **95**: Top 5% persistence scores are anomalies (recommended)
- **90**: Top 10% are anomalies (more sensitive)

---

## Output Files

### Intermediate Data Files

| File | Description |
|------|-------------|
| `data_pca.npy` | PCA-transformed data for each n_components |
| `windows_dict.npy` | Dictionary of sliding windows {n: {L: windows}} |
| `labels_raw.npy` | Original labels aligned with windows |
| `topo_features.npy` | TDA features for each window |
| `topo_scores.npy` | Max persistence scores |
| `topo_diagrams.npy` | Persistence diagrams |
| `ml_preds.npy` | ML model predictions |

### Results Files

| File | Description |
|------|-------------|
| `ml_param_log.csv` | Log of ML parameters tested |
| `ablation_results.csv` | All evaluation results |

### Visualization Files

| File | Description |
|------|-------------|
| `topo_ablation_heatmap.png` | Heatmap of F1-scores |
| `metric_comparison.png` | Bar chart of best F1 per model |
| `diagram_sidebyside.pdf` | Persistence diagram comparison |
| `analysis_notes.md` | Summary and analysis |

---

## Examples

### Example 1: Minimal Run

```bash
# Assuming data files are in data/ folder
python main.py --all
```

### Example 2: Custom Data Location

```bash
# Data is in a different folder
python main.py --data \
    --data-file /path/to/my/secom.data \
    --labels-file /path/to/my/labels.data

# Then continue with other steps
python main.py --tda --ml --eval --viz
```

### Example 3: Quick Experiment

```bash
# Test with fewer configurations
python main.py --data --pca 2 3 --windows 20 30
python main.py --tda
python main.py --ml
python main.py --eval
python main.py --viz
```

### Example 4: Resume from Checkpoint

```bash
# If data processing is done, skip it
python main.py --tda --ml --eval --viz
```

---

## Understanding the Results

### Key Metrics

- **F1-Score**: Harmonic mean of Precision and Recall (main metric)
- **AUC**: Area Under ROC Curve (overall discriminative ability)
- **Precision**: Of predicted anomalies, how many are true
- **Recall**: Of true anomalies, how many are detected

### Expected Best Configuration

Based on the original ablation study:
- **Topology**: PCA=2, L=30, Threshold=95th percentile
- **Best F1**: ~0.12-0.15
- **Note**: Low F1 is expected due to highly imbalanced data (~6.6% anomaly)

### Interpreting Heatmap

The ablation heatmap shows F1-scores for each (PCA, L) combination:
- **Higher values** (darker blue) = better performance
- **Optimal region** is typically PCA=2, L=20-30

### Interpreting Persistence Diagrams

- Points **far from diagonal** (y=x): Long-lived loops = significant features
- Points **near diagonal**: Short-lived loops = noise
- With larger L, points move closer to diagonal (features "diluted")

---

## Troubleshooting

### Common Issues

#### 1. File Not Found Error

```
FileNotFoundError: [Errno 2] No such file: 'data/secom.data'
```

**Solution**: Ensure data files are in the correct location or use `--data-file` and `--labels-file` flags.

#### 2. Memory Error

```
MemoryError: Unable to allocate array...
```

**Solution**: Reduce window sizes or PCA components:
```bash
python main.py --data --pca 2 --windows 20
```

#### 3. Ripser Not Available

```
ImportError: No module named 'ripser'
```

**Solution**: Install TDA dependencies:
```bash
pip install ripser persim
```

#### 4. WeasyPrint Error (PDF Generation)

If PDF generation fails, visualizations will still work and save as PNG files.

---

## Citation

If you use this code in your research, please cite:

```
SECOM TDA Project
Topological Data Analysis for Anomaly Detection in Semiconductor Manufacturing
```

---

## License

This project is for educational and research purposes.

---

## Contributors

| Name | Role | Contribution |
|------|------|--------------|
| Trần Thái Tuấn | Developer | Project development, TDA implementation, documentation |

---

## Contact

For questions or issues, please open an issue on the repository.

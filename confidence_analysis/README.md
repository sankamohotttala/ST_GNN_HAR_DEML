# 🧠 ST-GNN HAR Confidence Analysis

This repository contains code for analyzing **confidence values** in Human Activity Recognition (HAR) using **Spatial-Temporal Graph Neural Networks (ST-GNN)**.  
The project focuses on analyzing and comparing different confidence calculation methods for skeleton-based activity recognition.

---

## 📌 Project Overview

This project analyzes confidence values and classification accuracy for HAR using skeleton data.  
It implements multiple approaches for confidence calculation and provides comprehensive visualization tools for result analysis.

---

### ✨ Key Features

- 🔢 Multiple confidence calculation methods:
  - Default confidence analysis
  - Frame-based confidence analysis
  - Joint-based confidence analysis
  - Per-person skeleton analysis

- 📊 Visualization tools for:
  - Accuracy comparison
  - Confidence distribution
  - Skeleton visualization with confidence values

- 👥 Support for both single-person and two-person skeleton analysis  
- 📈 Comprehensive data analysis and result comparison

---

## 📁 Directory Structure

```
.
├── CSV/                          # Raw data and classification results
├── Data/                         # Input skeleton data
├── Results_save/                 # Generated results and visualizations
├── results_exp/                  # Experimental results
├── dataAnalysis_*.py             # Analysis scripts
├── boxPlot*.py                   # Visualization scripts
└── labelnamesIndex.txt           # Class label mappings
```

---

## 🔧 Key Components

### 🧪 Data Analysis Scripts

1. **dataAnalysis_confidenceDefault.py**  
   - Main implementation of default confidence calculation  
   - Processes skeleton data and computes confidence values  
   - Generates basic confidence analysis results

2. **dataAnalysis_confidenceCompJoint.py**  
   - Joint-based confidence analysis  
   - Supports both single-person and two-person skeleton analysis  
   - Generates detailed joint-wise confidence values

3. **dataAnalysis_confidencePerPersonDef.py**  
   - Person-specific confidence analysis  
   - Analyzes confidence values per person in multi-person activities

---

### 📊 Visualization Scripts

1. **boxPlotAllApproaches.py**  
   - Generates box plots comparing different confidence calculation methods  
   - Supports visualization of:
     - All data
     - Correctly classified cases
     - Misclassified cases
     - Method comparisons

2. **confidencePlot.py**  
   - Visualizes confidence distributions  
   - Generates confidence-related plots

---

### 📂 Data Files

- **CSV Files** – Contains classification results and confidence values  
- **labelnamesIndex.txt** – Maps class indices to activity names  
- **Results_save/** – Directory storing generated visualizations

---

## ▶️ Usage

### 🔹 Basic Confidence Analysis
```bash
python dataAnalysis_confidenceDefault.py
```

### 🔹 Joint-based Analysis
```bash
python dataAnalysis_confidenceCompJoint.py
```

### 🔹 Visualization
```bash
python boxPlotAllApproaches.py
```

---

## 📄 Data Format

### Input
- Skeleton data in `.npy` (NumPy) format  
- Labels in `.pkl` (Pickle) format  
- CSV files with classification results

### Output
- CSV files with confidence analysis results  
- PNG files with visualizations  
- Analysis reports and charts

---

## 📊 Results

### 1. Confidence Values
- Per-class confidence distributions  
- Joint-wise confidence analysis  
- Person-specific confidence values

### 2. Visualizations
- Box plots comparing methods  
- Accuracy distributions  
- Skeleton visualizations with confidence overlays

### 3. Analysis Reports
- Per-class accuracy  
- Confidence-accuracy correlation  
- Statistical comparison of methods

---

## 🧩 Dependencies

- Python 3.x  
- NumPy  
- Pandas  
- Matplotlib  
- Seaborn  
- Pickle

---

## 🤝 Contributing

Feel free to submit issues and enhancement requests.

---

## 📜 License

[Specify your license here]

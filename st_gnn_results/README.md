# 📊 ST-GNN Results

This repository contains scripts, data, and results related to the evaluation of **spatio-temporal graph neural networks (ST-GNN)** for **human activity recognition (HAR)**. The directory is organized into subdirectories and files for different models, experiments, and utilities.

---

## 📁 Directory Overview

### 🔑 Key Files and Directories

#### `conf_create.py`
- 📌 Generates and saves confusion matrices (normalized and default) as heatmaps.
- 📥 Input: CSV file with real and predicted labels.
- 📤 Output:
  - `<trial>_conf_norm.jpg`: Normalized confusion matrix.
  - `<trial>_conf_default.jpg`: Default confusion matrix.

#### `test_2sagcn_scores.py`
- 📌 Processes and evaluates test scores for the `2s-AGCN` model.
- 📥 Input: Pickle files with test scores.
- 📤 Output: Printed results of score evaluation.

#### `z_ensemble_STGNN/`
- 📌 Scripts for ensemble model evaluation.
  - `ensemble_aagcn.py`: Combines predictions from multiple models.
  - `ensemble_save.py`: Saves ensemble scores and metrics.

#### `results/`
- 📌 Stores confusion matrices and evaluation outputs.

#### Model-specific Folders
- `2s-AGCN/`, `MS-AAGCN/`, `RA-GCN/`, `ST-GCN/`, `STGAT/`:  
  Each contains results and intermediate files for trials like `D`, `F`, `S`, `Sh`.

#### `ra-gcn full temp/`
- 📌 Contains final results for RA-GCN model:
  - `epoch30_test_score.pkl`: Final test scores.
  - `full_conf_default.jpg` & `full_conf_norm.jpg`: Confusion matrices.
  - `full_epoch30.csv`: Real vs. predicted labels.

#### Miscellaneous
- `ensemble_scores.pkl`, `epoch26_test_score.pkl`, `epoch30_test_score.pkl`:  
  Pickle files with test/ensemble scores.
- `Figure_1.png`: A figure summarizing evaluation or results.
- `readme.txt`: Additional notes or instructions.

---

## ⚙️ How to Use

### 1. Generate Confusion Matrices
- Run `conf_create.py`.
- Set appropriate `file_path` and `trial` values.

### 2. Evaluate Ensemble Models
- Run `z_ensemble_STGNN/ensemble_aagcn.py` and `ensemble_save.py`.

### 3. View Results
- Browse the `results/` folder for all output visualizations.

---

## 📦 Requirements

- Python 3.x
- Required Libraries:
  - `numpy`, `pandas`, `matplotlib`, `seaborn`, `pickle`, `tqdm`

---

## 📄 License

This project is licensed under the **MIT License**.  
See the `LICENSE` file for full details.

---

## 🙏 Acknowledgments

- This work is part of research on ST-GNNs for HAR.
- Thanks to all contributors and maintainers of the models and datasets used.

# 📁 KS-KSS Results

This repository contains selected output files and visualizations from experiments conducted on the **KS (Kinetics Skeleton)** and **KSS (Kinetics Skeleton Subset)** datasets for child action recognition tasks. The focus is on model evaluations using different protocol settings, transfer learning configurations, and visual analytics of the results.

---

## 📂 Repository Structure

```
.
├── ks/                            
├── kss/                           
├── other_ks_kss_results/         
├── README.md                     
├── [original]results (version 1).xlsx   
├── ks_hyperparameters_and_other_details.xlsx
```

---

## 📁 kss/

This folder includes KSS dataset-based implementation results. Only selected outputs are provided here.

### 🔹 Protocol Subfolders

- `3 class/`
- `5 - balanced class/`
- `5 - unbalanced class/`
- `8 class/`

Each protocol folder contains:

- **FT**, **FX**, **Prop**, **vanilla** — representing different model configurations:
  - `vanilla`: Baseline implementation
  - `FT`: Fine-tuning based Transfer Learning (TFL)
  - `FX`: Feature extraction-based TFL
  - `Prop`: Propagation method-based TFL

### 📊 Contents per Configuration

- 📈 Training loss and accuracy plots  
- 📅 Epoch-wise accuracy values  
- 🔀 Confusion matrices  
- 📊 Dataset distribution plots  
- 📄 CSV files with final test dataset softmax probability values

### 📄 Supplementary Files

- **TFL results.docx** – Summarized TFL experiment results and box-and-whisker plots  
- **visualizaation stuff.docx** – Skeleton visualizations and KS-KSS dataset distribution data

---

## 📁 ks/

This folder contains results for KS dataset-based implementations. It has two main folders:

- `vanilla/`: Results from models trained from scratch  
- `TFL/`: Results from transfer learning experiments  

Each of these contains protocol folders:

- `3 classes/`  
- `5 classes/`  
- `5 classes - balance/`  
- `8 classes/`

Each result directory includes:

- 🧩 `images_independant/` – Confusion matrices for all epochs  
- 📜 `log/` – Python scripts, logs, distributions, hyperparameters  
- 📉 `matplotlib_graphs/` – Loss and accuracy plots  
- 📄 `save_values_csv/` – Softmax test-set probability values in CSV format

---

## 📁 other_ks_kss_results/

This folder holds sub-optimal or intermediate results from various KS experiments (both vanilla and TFL). Each subfolder represents a distinct implementation configuration.

### 📂 Folder Structure

- `images_independant/`  
- `log/`  
- `matplotlib_graphs/`  
- `save_values_csv/`

🧾 **Refer to `ks_hyperparameters_and_other_details.xlsx`** to match folder names with their hyperparameter configurations. These results help analyze how different settings impact child action recognition, especially in in-the-wild scenarios.

---

## 📄 Other Files

- **ks_hyperparameters_and_other_details.xlsx**  
  - Contains hyperparameter configurations and experiment notes  
  - Rows tagged as `"journal"` (highlighted in yellow) refer to final results used in publications  
  - Rows tagged as `"added"` (highlighted in orange) refer to additional results in `other_ks_kss_results`

- **[original]results (version 1).xlsx**  
  - Legacy version of the hyperparameter and observation tracking sheet

---

## 📌 Summary

This repository supports structured exploration and comparison of KS and KSS-based child action recognition models. It provides insights into performance across different configurations and helps in evaluating the effect of hyperparameters in both ideal and real-world conditions.

# Transfer Learning Results for ST-GCN based CAR

This directory contains experiment results and supporting code related to transfer learning (TFL) applied to child action recognition (CAR) using the ST-GCN (Spatio-Temporal Graph Convolutional Network) model. The experiments are based on various fine-tuning configurations using the CWBG dataset.

## Folder Structure

### 🔬 Experiment Result Folders

- **FT/**\
  Contains results where the model was fine-tuned as detailed in the journal paper.

- **FT-random/**\
  Contains results where the model was trained with hybrid-frozen methods.

- **FX/**\
  Contains experiments  of feature extraction experiments.

Each of these folders includes:

- Result logs
- Associated visualizations
- A `log/` subfolder containing the exact `.py` files used in that experiment along with final hyper-parameter settings.

### 🧠 Supporting Code Folders

- **graph/**\
  Utilities and graph configuration files used in model input preparation.

- **model/**\
  Contains architecture definitions and neural components relevant to TFL experiments.

---

## Code Files

The Python scripts provided are part of the experiment pipeline, but note that **not all scripts used in experiments are listed here**.  
To find the exact scripts used in a specific experiment, refer to the `log/` folder within that experiment’s result directory, where the used `.py` files and corresponding hyper-parameter settings are preserved.

- `cwbg_tf_FX_layers.py`, `cwbg_tf_FX_layers_loop.py`, `cwbg_tf_FX_layers_testonly.py`  
  Scripts for running various configurations of feature extraction (FX) mode experiments.

- `main_cwbg.py`, `main_tf_default_for_all_layers.py`, `main_tf_default_random.py`, `main_tf_FX_layers.py`  
  Main scripts for executing full model fine-tuning, hybrid training, and default TFL modes.

- `tools.py`, `utility_custom.py`, `weight_save.py`  
  Supporting scripts used across experiments for utilities, saving model weights, and custom components.

- `get_statistics.py`  
  Script for extracting performance metrics and summarizing experiment results.

---

## Additional Files

- `implementations FX FT and other such - LOOP.xlsx`  
  A spreadsheet documenting different TFL configurations and key observations from the experiments. Use it to trace which setup was used under each folder and what changes were tested.

- `readme.txt`  
  For internal contributor reference only. 

---

## Notes

- 📂 All experiments are based on the **CWBG** dataset and were designed around transfer learning principles for child action recognition.
- 🔍 For reproducibility, always check the `log/` folder in each experiment folder to find the exact Python scripts used during the final run.

---



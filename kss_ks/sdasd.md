# ks_kss_results

This repository contains selected output files and visualizations from experiments conducted on the KS (Kinetics Skeleton) and KSS (Kinetics Skeleton Subset) datasets for child action recognition tasks. The focus is on model evaluations using different protocol settings, transfer learning configurations, and visual analytics of the results.

## Repository Structure

```
.
├── ks/                            
├── kss/                           
├── other_ks_kss_results/         
├── README.md                     
├── [original]results (version 1).xlsx   
├── ks_hyperparameters_and_other_details.xlsx
```

### kss/

This folder includes KSS dataset-based implementation results. Only selected outputs are provided here. The folder contains subfolders for each experimental protocol:

#### Folders

- `3 class/`
- `5 - balanced class/`
- `5 - unbalanced class/`
- `8 class/`

Each protocol folder contains:

- **FT**, **FX**, **Prop**, **vanilla** — representing different model configurations:
  - `vanilla`: Baseline implementation
  - `FT`: Fine-tuning based transfer learning (TFL)
  - `FX`: Feature extraction-based TFL
  - `Prop`: Propagation method-based TFL 

Within each configuration folder, the following results are included:

- **Training loss and accuracy plots**
- **Epoch-wise accuracy values**
- **Confusion matrices**
- **Dataset distribution plots**
- **CSV files** with final test dataset softmax probability values

#### Files

- **TFL results.docx**  
  Contains summarized results of transfer learning experiments along with box-and-whisker plots.

- **visualizaation stuff.docx**  
  Includes skeleton visualizations and KS-KSS dataset distribution data.

### ks/

This folder contains results for KS dataset-based implementations. It has two main folders:

- `vanilla/`: Results from baseline models trained from scratch.
- `TFL/`: Results from transfer learning-based experiments.

Each of these contains four protocol subfolders:

- `3 classes/`
- `5 classes/`
- `5 classes - balance/`
- `8 classes/`

Under each protocol, only the final used (best-performing) implementation results are included. Each result directory contains:

- `images_independant/`: Confusion matrices for all epochs.
- `log/`: Python experiment scripts, logs, dataset distributions, and hyperparameter settings.
- `matplotlib_graphs/`: Loss and accuracy visualizations.
- `save_values_csv/`: Softmax test-set probability values for most epochs in CSV format.

### Other Files

- **[original]results (version 1).xlsx**  
  Early version of the complete result compilation.

- **ks_hyperparameters_and_other_details.xlsx**  
  Provides dataset descriptions, protocol definitions, and hyperparameter configurations for both KS and KSS datasets.

---

This repository is intended for structured exploration and comparison of KS and KSS-based action recognition models, highlighting performance across multiple configurations and protocols.

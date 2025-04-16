# KS-KSS Results

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

### other_ks_kss_results/

This folder contains sub-optimal or experimental results from various KS implementations, covering both vanilla and transfer learning approaches. Each subfolder represents a unique configuration or experimental run. The naming format of the folder corresponds to specific hyperparameter settings.

- The structure of each folder includes:
  - `images_independant/`
  - `log/`
  - `matplotlib_graphs/`
  - `save_values_csv/`

- Additional details related to each implementation can be found in the file **ks_hyperparameters_and_other_details.xlsx**. Use the folder name to locate the corresponding row in the spreadsheet.

These results can be useful for understanding how different hyperparameter configurations affect performance, especially when evaluating in-the-wild scenarios involving child action recognition.

### Other Files

- **ks_hyperparameters_and_other_details.xlsx**  
  Provides hyperparameter configurations and observation made by the researchers during the KS experiments. First column relates to the original experiment results folder in the server. Final experiments used in the Journal paper are mentioned as "journal" and they are highlighted in yellow colour. Results added in other_ks_kss_results folder are also mentioned as "added" and highlighted in orange colour.

- **[original]results (version 1).xlsx**  
  Original version of the ks_hyperparameters_and_other_details.xlsx file.


---

This repository is intended for structured exploration and comparison of KS and KSS-based action recognition models, highlighting performance across multiple configurations and protocols.

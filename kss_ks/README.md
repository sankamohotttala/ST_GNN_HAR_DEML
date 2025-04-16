# ks_kss_results

This repository contains selected output files and visualizations from experiments conducted on the KS (Kinetics Skeleton) and KSS (Kinetics Skeleton Subset) datasets for human action recognition tasks. The focus is on model evaluations using different protocol settings, transfer learning configurations, and visual analytics of the results.

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

- `3 class/`
- `5 - balanced class/`
- `5 - unbalanced class/`
- `8 class/`

Each protocol folder contains:

- **FT**, **FX**, **Prop**, **vanilla** — representing different model configurations:
  - `vanilla`: Baseline implementation
  - `FT`: Fine-tuned models
  - `FX`: Feature extraction-based transfer learning
  - `Prop`: Proposed model variants

Within each configuration folder, the following results are included:

- Training **loss and accuracy plots**
- **Epoch-wise accuracy values**
- **Confusion matrices**
- **Dataset distribution plots**
- **CSV files** with final test dataset softmax probability values

### Summary Files

- **TFL results.docx**  
  Contains summarized results of transfer learning experiments.

- **visualizaation stuff.docx**  
  Includes skeleton visualizations and box-and-whisker plots for softmax-based output distributions.

### Other Files

- **[original]results (version 1).xlsx**  
  Early version of the complete result compilation.

- **ks_hyperparameters_and_other_details.xlsx**  
  Provides dataset descriptions, protocol definitions, and hyperparameter configurations for both KS and KSS datasets.

---

This repository is intended for structured exploration and comparison of KS and KSS-based action recognition models, highlighting performance across multiple configurations and protocols.

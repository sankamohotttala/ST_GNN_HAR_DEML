# Kinetics Skeleton Reproducing Experiments

This directory contains various experimental results and logs related to reproducing and extending the ST-GCN model on the Kinetics dataset, particularly focusing on the *kinetics-motion* and *kinetics-392* subsets.

## 📁 Directory Structure

Each folder (e.g., `0_30_24503_kinetics_motion_30_From_Scratch...`) corresponds to a unique experiment with a different configuration or subset of the dataset. Inside each experiment folder, the following structure is typically found:

```
experiment_folder/
├── images_independant/      # Sample output visualizations
├── log/                     # Training logs and .py files used for that run
├── matplotlib_graphs/       # Accuracy/loss curves and other plots
└── save_values_csv/         # Inference time and other saved numerical results
```

These folders provide a comprehensive view of each experiment's output and performance.

## 📊 Experiment Details

Some of the experiments included here are based on variations in:

- **Subset Size / Sampling Strategy:** Full dataset vs. sampled subsets
- **Training from Scratch vs. Pre-trained:** Most models are trained from scratch unless stated
- **Hyperparameters:** Learning rate, optimizer, scheduler, and batch size

Details of selected experiments are documented in the Excel file:
```
[kinetics-skeleton]implem_details_ST-GCN.xlsx
```
This file includes the following for each experiment:
- Epochs
- Optimizer
- Learning Rate
- Batch Size
- Scheduler Type
- Observations and Notes (including train/test accuracy and qualitative comments)
- Dataset type (e.g., `kinetics-392`, `kinetics-motion`)
- Whether a pre-trained model was used

You may use the folder name (e.g., `24503`, `234951`) to match rows in the Excel file.

## 📌 Notes

- Experiments labeled with "From_Scratch" indicate no pre-trained weights were used.
- Some implementations explore sub-optimal or alternate hyperparameter settings.
- These variations help assess the reproducibility of the original ST-GCN results and the sensitivity of the model to different training configurations.

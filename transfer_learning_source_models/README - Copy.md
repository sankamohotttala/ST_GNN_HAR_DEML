# Transfer Learning Source Models

This directory contains source models and related experiment files for the transfer learning experiments described in our journal paper. The provided models are based on several splits of the NTU dataset, including NTU-120, NTU-60, NTU-5, NTU-44-B, NTU-44-W, and NTU-60-FRA.

## Folder Structure

Each subfolder (e.g., `ntu120`, `ntu60`, `ntu5`, etc.) contains files and directories relevant to that specific dataset split. Typical contents include:

- **config.yaml**  
  Contains the hyperparameter details used for the experiment, such as number of epochs, batch size, learning rate, etc.

- **events.out.tfevents.{unique_val}.0.v2**  
  TensorBoard event file that logs training and evaluation metrics.

- **stgcn.py**  
  The ST-GCN model architecture used for the source model training.

- **weight-checkpoint/**  
  Directory containing the selected trained model weights. These weights correspond to the Top-1 and Top-5 accuracy values reported in the source model results table in the journal paper.

- **independant images/**  
  Contains confusion matrix images generated for each epoch during training.

- **log/**  
  Contains backups of important files for reproducibility, such as `config.yaml`, `stgcn.py`, and the event file, along with additional logs.

- **matplotlib_graphs/**  
  Visualizations generated during the experiment, including plots (accuracy/loss curves, etc.) and t-SNE visualizations.

- **save_values_csv/**  
  Directory with CSV files containing saved metric values (softmax probability etc.) for each epoch.

## Availability

- This repository contains models and results only for the following NTU dataset splits:
  - NTU-120
  - NTU-60
  - NTU-5
  - NTU-44b
  - NTU-44w
  - NTU-60fra

  Model weights and results for other dataset splits or protocols can be shared upon request. Some of these are already available on Google Drive.

- **Note:** Only some event files are included in this repository due to size limitations.

## Accessing Full Results

To access the complete set of results, including all checkpoints, confusion matrices, and event files for most source models, please refer to the following Google Drive link:

[Google Drive - Full Results and Checkpoints](https://drive.google.com/drive/folders/1fY6NaHDik52XW6KmZnEX1bOOWsQjqQBs?usp=sharing)


This link includes:
- All checkpoints for most source models
- Additional result files, including confusion matrices and event logs


---

## Additional Notes

- The `readme.txt` file is for repository contributors and contains internal notes.
- For experiment-specific code files, refer to the `log` folder within each experiment results directory; all Python scripts used (with their final hyper-parameters) are stored there.

---


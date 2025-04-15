# CWBG Vanilla Implementations

This README provides a brief description of the files and directories included in this project for vanilla CWBG implementations.

## 📁 Directories

- **graph/**  
  Contains graph-related data structures, adjacency matrices, or utilities used for processing graph inputs.

- **model/**  
  Contains STGCN model architecture used in vanilla implementations. Regularizer and weight initialization related parameters are given in these files.

- **results/**  
  Includes evaluation outputs for all CWBG vanilla protocols. Due to storage limitations only some of the results are given here.  
  To access the full results, download the zip file from this link:  
  [Google Drive - Full Results (~4GB)](https://drive.google.com/drive/folders/1STHC01cJjZuXfCfTa13R_mQxmx4pO6KB?usp=sharing)

  This folder contains:
  - `proto_cwbg`: Results related to prototype-based CWBG protocol.
  - `random_cwbg`: Evaluation results using random splits for CWBG.
  - `loocv_cwbg`: Outputs obtained via leave-one-out cross-validation for CWBG.

- **saved_weights/**  
  Includes the saved checkpoints for some models. Due to storage limitations, only a few of the final model weights are added here.  
  To access all checkpoints for multiple protocols, use this link:  
  [Google Drive - Model Checkpoints](https://drive.google.com/drive/folders/1GNotfE_50zuxGQ0d91AaXF2PoXXf0nzF?usp=sharing)
  This folder contains:
    - `0_10_628_cwbg_dissimilar_30_from_scratch_4_2023_02_21__23_58_53_False_0.01_[10, 20]`: This contains the first first checkpoint (2nd epoch) and final checkpoint saved after the final epoch (epoch 30th). These model weights are for the implementation in the path "\vanilla_cwbg\results\cross_subject\cwbg_dissimilar\0_10_628_cwbg_dissimilar_30_from_scratch_4_2023_02_21__23_58_53_False_0.01_[10, 20]".


## 📄 Python Scripts

- **get_statistics.py**  
  Script to compute average accuracy and standard deviation for repeated hold-out validations.

- **main_test.py**  
  Main code implementation for CWBG vanilla with cross-subject protocol. Out of the 30 subjects, 21 (70%) were used for training set and rest for testing.

- **main_test_LOOCV.py**  
  Main code implementation for CWBG vanilla with leave-one-out-cross-validation (LOOCV) where one subject was used as the leaved subject and the testing was done on all the actions of that subject. Results from these implementations are used for age-wise and gender-wise analysis as well.

- **main_test_newProto_random.py**  
  Main code implementation for CWBG vanilla with random-split protocol.

- **utility_custom.py**  
  Contains custom utility functions used in the main scripts.

- **smaller_folder_create.py**  
  This file was used to create a smaller size results folder for each implementation across each protocol so that the essential results are on this git repository without exceeding the storage limitations of GitHub. In this process, only the final confusion matrix, final results csv file, final plots were added to this repository. To access all results used the above given Google Drive link.

- **time_cal.py**  
  This script was used to calculate the average inference time and teh standard deviation of the  inferene time across all protocols. 


## 📄 Other files

- **cwbg_vanilla_selected_hyperparams.csv**
  Contains the hyper-parameters and other user parameters used across all protocols.  

  


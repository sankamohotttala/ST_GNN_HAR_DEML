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

## 📄 Python Scripts

- **get_statistics.py**  
  Script to compute various performance statistics such as accuracy, precision, recall, or other custom metrics.

- **main_test.py**  
  Main testing script for evaluating the model using a predefined dataset split (likely train/test or validation).

- **main_test_LOOCV.py**  
  Variant of the test script implementing Leave-One-Out Cross-Validation (LOOCV) for performance evaluation.

- **main_test_newProto_random.py**  
  Another test variation using a new prototype-based random sampling method, possibly for few-shot or prototypical learning tasks.

- **utility_custom.py**  
  Contains custom utility functions used across scripts such as data loading, preprocessing, or helper functions.

---
# Data-Efficient Spatio-Temporal Graph Neural Network based Child Action Recognition: A Systematic Analysis

This repository contains TensorFlow-based **ST-GNN** codes that can be used for **child action recognition**.  
Some pre-processed **CWBG datasets** are stored on Google Drive and can be accessed via the links provided below.

---

##  CWBG Datasets

### ➤ Skeleton-Formatted CWBG Dataset
-  [Skeleton-Formatted CWBG Dataset](https://drive.google.com/drive/folders/1v1v1EP2NKSMPrzxHmH7CksS9r2Qu_OO2?usp=drive_link)  
  Includes pose-estimated skeleton data of children in `.npy` format ready to be used for model training.

---

### Processed CWBG Dataset (All Protocols)

- [Processed CWBG Dataset Folder](https://drive.google.com/drive/folders/1XlNMdLMFJkSPCTFSC03TsxNqXWrt-6bS?usp=sharing)  
  This directory contains fully preprocessed data for all three evaluation protocols: **Cross-Subject**, **Random Split**, and **LOOCV**.

Each folder within this dataset contains:

- `.npy` files (e.g., `train_data_joint.npy`, `val_data_joint.npy`): These are the output of the `gen_joint_data.py` script, which parses raw skeleton text files, extracts 3D joint positions using `read_xyz()`, applies normalization (`pre_normalization()` from `preprocess.py`), and saves the data in a standardized format.
  
- `.pkl` files (e.g., `train_label.pkl`, `val_label.pkl`): Pickled Python lists containing sample names and their corresponding class labels.

- `.tfrecord` files (e.g., `train_data-0.tfrecord`, ...): These are generated using `gen_tfrecord_data.py`, which converts `.npy` and `.pkl` files into TensorFlow’s efficient binary TFRecord format using serialized examples (`serialize_example()`).

Below is the file structure for the preprocessing code directory:

```
data_processing/data_pre_processing/
├── __init__.py
├── gen_joint_data.py
├── gen_joint_data_test.py
├── gen_tfrecord_data.py
├── preprocess.py
├── preprocess_no_edit.py
├── readme.txt
├── rotation.py
```

Refer to the following scripts for details on the data pipeline:
- `preprocess.py`: Skeleton alignment, padding, and canonicalization
- `gen_joint_data.py`: Joint data extraction and preprocessing
- `gen_tfrecord_data.py`: TFRecord conversion from `.npy` and `.pkl` formats
- `rotation.py`: Utility functions for vector-based rotation and coordinate normalization

This setup ensures efficient training and evaluation by standardizing the skeleton sequences and enabling high-performance data loading in TensorFlow.


#### Cross-Subject Protocol

- [CWBG-Full](https://drive.google.com/drive/folders/1T9kgWkrNlrPm_eKbY3NfBXsGVLDdBPt-?usp=share_link): Contains the full dataset with 1312 skeleton sequences across 15 classes.
- [CWBG-Dissimilar](https://drive.google.com/drive/folders/1TwUnf5G_4IhLIh04Q1vb-JGPt1G5Hfby?usp=share_link): Contains the CWBG Dissimilar subset with 10 classes.
- [CWBG-Similar](https://drive.google.com/drive/folders/1RUymfektG0jyCCpRr5Mdw-eWt2Pcxzzt?usp=share_link): Contains the CWBG Similar subset with 10 classes.
- [CWBG-Shared](https://drive.google.com/drive/folders/1RMKR7cxV7BTCTjUTBSTAhxwXscBHeFFB?usp=share_link): Contains the CWBG Shared subset with 5 classes.


#### Random Split Protocol
-  [Full Dataset Zip file](https://drive.google.com/file/d/1a80YigS6b1lG-uZIekOq1mr1o8_X7Xq6/view?usp=drive_link)  
  This folder contains pre-processed datasets categorized under multiple evaluation protocols.

####  LOOCV Protocol
-  [Full Dataset Zip file](https://drive.google.com/file/d/1NRsLw8au6o5YgO3lg9p09kj8TCWmjjyR/view?usp=drive_link)  
  This folder contains pre-processed datasets categorized under multiple evaluation protocols.


**Folder Structure:**
```
Processed_CWBG_Dataset/
├── Cross_Subject/
│   ├── CWBG-Full/
│   ├── CWBG-Dissimilar/
│   ├── CWBG-Similar/
│   └── CWBG-Shared/
│
├── Random_Split/
│   ├── CWBG-Full/
│   ├── CWBG-Dissimilar/
│   ├── CWBG-Similar/
│   └── CWBG-Shared/
│
└── LOOCV/
    ├── CWBG-Full/
    ├── CWBG-Dissimilar/
    ├── CWBG-Similar/
    └── CWBG-Shared/
```

 For further protocol-specific details and implementation instructions, refer to:
- [`README_cwbg.md`](./README_cwbg.md)

---

## Usage

To use the datasets with the ST-GCN code:

- Modify **lines 415–426** in the `./basic_code/main.py` file to include the **absolute paths** to your datasets.
- Use the **visualization function** if needed.

---

## Contact

For any further clarification regarding the use of the code or datasets, please reach out to:

- sanka.m@sliit.lk  
- divandyasm@gmail.com

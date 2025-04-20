# Data-Efficient Spatio-Temporal Graph Neural Network based Child Action Recognition: A Systematic Analysis

This repository contains TensorFlow-based **ST-GNN** codes that can be used for **child action recognition**.  
Some pre-processed **CWBG datasets** are stored on Google Drive and can be accessed via the links provided below.

---

##  Datasets

### ➤ Skeleton-Formatted CWBG Dataset
-  [Skeleton-Formatted CWBG Dataset](https://drive.google.com/drive/folders/1v1v1EP2NKSMPrzxHmH7CksS9r2Qu_OO2?usp=drive_link)  
  Includes pose-estimated skeleton data of children in `.npy` format ready to be used for model training.

### ➤ Processed CWBG Dataset (All Protocols)
-  [Processed CWBG Dataset Folder](https://drive.google.com/drive/folders/1OnmErZipnys0SDTStwZgXB3eSRHH9ZKr?usp=sharing)  
  This folder contains pre-processed datasets categorized under multiple evaluation protocols.


#### Cross-Subject Protocol

- [CWBG-Full](https://drive.google.com/drive/folders/1T9kgWkrNlrPm_eKbY3NfBXsGVLDdBPt-?usp=share_link): Contains the full dataset with 1312 skeleton sequences across 15 classes.
- [CWBG-Dissimilar](https://drive.google.com/drive/folders/1TwUnf5G_4IhLIh04Q1vb-JGPt1G5Hfby?usp=share_link): Contains the CWBG Dissimilar subset with 10 classes.
- [CWBG-Similar](https://drive.google.com/drive/folders/1RUymfektG0jyCCpRr5Mdw-eWt2Pcxzzt?usp=share_link): Contains the CWBG Similar subset with 10 classes.
- [CWBG-Shared](https://drive.google.com/drive/folders/1RMKR7cxV7BTCTjUTBSTAhxwXscBHeFFB?usp=share_link): Contains the CWBG Shared subset with 5 classes.


#### Random Split Protocol
- Refer to `random/` folder inside results or the structure noted in `README_cwbg.md`.

####  LOOCV Protocol
- Refer to `loocv/` folder inside results or the structure noted in `README_cwbg.md`.

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

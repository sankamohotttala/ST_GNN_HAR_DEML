# ST_GNN_HAR_DEML
<!-- This contains the codes used in human action recognition tasks with small datasets.
 -->
 This repository contains tensorflow based ST-GCN code that can be used for child action recognition. Pre-processed CWBG datasets are stored on Google Drive and can be accessed via the links provided below.

## Datasets

1. [CWBG Full Dataset](https://drive.google.com/drive/folders/1T9kgWkrNlrPm_eKbY3NfBXsGVLDdBPt-?usp=share_link) - Contains the full dataset with 1312 skeleton sequences in 15 classes.
2. [CWBG Dissimilar Dataset](https://drive.google.com/drive/folders/1TwUnf5G_4IhLIh04Q1vb-JGPt1G5Hfby?usp=share_link) - Contains the CWBG Dissimilar dataset/subset with 10 classes.
3. [CWBG Similar Dataset](https://drive.google.com/drive/folders/1RUymfektG0jyCCpRr5Mdw-eWt2Pcxzzt?usp=share_link) - Contains the CWBG Similar dataset/subset with 10 classes.
4. [CWBG Shared Dataset](https://drive.google.com/drive/folders/1RMKR7cxV7BTCTjUTBSTAhxwXscBHeFFB?usp=share_link) - Contains the CWBG Shared dataset/subset with 5 classes.

## Usage

To use the datasets with ST-GCN code, change the 415-426 lines in the ./ST-GCN code/main.py file so that they contain the absolute paths for datasets. Use the visualization function if needed. 

## Contact Details
For any further clarification regarding the use of code/dataset, please reach via sanka.m@sliit.lk or divandyasm@gmail.com.

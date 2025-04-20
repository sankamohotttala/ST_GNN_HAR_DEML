Data-Efficient Spatio-Temporal Graph Neural Network based Child Action Recognition: A Systematic Analysis

This repository contains TensorFlow-based **ST-GCN** code that can be used for **child action recognition**.  
Pre-processed **CWBG datasets** are stored on Google Drive and can be accessed via the links provided below.

---

## Datasets

1. [CWBG Full Dataset](https://drive.google.com/drive/folders/1T9kgWkrNlrPm_eKbY3NfBXsGVLDdBPt-?usp=share_link)  
   Contains the full dataset with 1312 skeleton sequences in 15 classes.

2. [CWBG Dissimilar Dataset](https://drive.google.com/drive/folders/1TwUnf5G_4IhLIh04Q1vb-JGPt1G5Hfby?usp=share_link)  
   Contains the CWBG Dissimilar subset with 10 classes.

3. [CWBG Similar Dataset](https://drive.google.com/drive/folders/1RUymfektG0jyCCpRr5Mdw-eWt2Pcxzzt?usp=share_link)  
   Contains the CWBG Similar subset with 10 classes.

4. [CWBG Shared Dataset](https://drive.google.com/drive/folders/1RMKR7cxV7BTCTjUTBSTAhxwXscBHeFFB?usp=share_link)  
   Contains the CWBG Shared subset with 5 classes.

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

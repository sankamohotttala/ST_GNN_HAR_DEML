# Data-Efficient Spatio-Temporal Graph Neural Network based Child Action Recognition: A Systematic Analysis

This repository contains TensorFlow-based implementations of **Spatio-Temporal Graph Neural Networks (ST-GNNs)** developed as part of the journal paper titled:

> *"Data-Efficient Spatio-Temporal Graph Neural Network based Child Action Recognition: A Systematic Analysis"*

The manuscript is currently under review at a Q1-ranked Elsevier journal. This repository includes the core model implementations, data preprocessing pipelines, the majority of experimental results, and detailed observations made by the researchers during experimentation. Additionally, many of the hyperparameter settings, pre-processed data under each protocol and the corresponding saved model weights are provided. Together, these resources are intended to facilitate full reproducibility of the results presented in the journal paper.

The work focuses on **child action recognition** using pose-based features extracted from the **Child-Whole Body Gesture (CWBG)** dataset. The original CWBG dataset can be accessed via the official source:

🔗 [CWBG Official Dataset Link](http://www.eed.usv.ro/~vatavu/projects/DissimilarityConsensus/)

---

##  CWBG Datasets

CWBG dataset is under the `cwbg_dataset/` directory. This dataset is used for the in-the-lab experiments of child action recognition with ST-GNN models.


```
cwbg_dataset/
├── skeleton format/
│   ├── CWBG_skeleton_format.zip      # Raw skeleton data in .skeleton format
│   └── S001C001P001R001A001.skeleton # Example skeleton file
│
└── final train-test split/
    ├── cwbg_cross_subject.zip        # Cross-Subject protocol
    ├── for_LOOCV.zip                 # Leave-One-Out Cross Validation (LOOCV) protocol
    └── random_split.zip              # Random Split protocol
```


Each folder within this dataset contains:

- `.npy` files (e.g., `train_data_joint.npy`, `val_data_joint.npy`): These are the output of the `gen_joint_data.py` script, which parses raw skeleton text files, extracts 3D joint positions using `read_xyz()`, applies normalization (`pre_normalization()` from `preprocess.py`), and saves the data in a standardized format.
  
- `.pkl` files (e.g., `train_label.pkl`, `val_label.pkl`): Pickled Python lists containing sample names and their corresponding class labels.

- `.tfrecord` files (e.g., `train_data-0.tfrecord`, ...): These are generated using `gen_tfrecord_data.py`, which converts `.npy` and `.pkl` files into TensorFlow’s efficient binary TFRecord format using serialized examples (`serialize_example()`).

- `.pkl` files (e.g., `train_label.pkl`, `val_label.pkl`): Pickled Python lists containing sample names and their corresponding class labels.

- `.skeleton` files (e.g., `S001C001P001R001A001.skeleton`): contains the NTU RGB+D dataset format converted version of the CWBG dataset. These `.skeleton` files are processed using scripts in `data_processing/data_pre_processing/` to produce:
	- `.npy` files (joint arrays for model training)
	- `.pkl` files (labels and metadata)
	- `.tfrecord` files (TensorFlow-compatible formats)

---

### Data Protocols

The CWBG dataset is organized into the following subsets:

- **CWBG-Full**: Complete dataset with 1312 sequences across 15 classes.
- **CWBG-Dissimilar**: 10-class subset with dissimilar actions.
- **CWBG-Similar**: 10-class subset with visually similar actions.
- **CWBG-Shared**: 5-class subset with overlapping actions.

---

### Evaluation Protocols

#### Cross-Subject Protocol
- Subjects 1–21 used for training; others for testing.
- Follows the protocol style of NTU RGB+D for cross-subject generalization.

#### Random Split Protocol
- Data is randomly divided into training and validation sets with class balance.

#### LOOCV Protocol
- Each subject is left out once as the test set.
- Ensures robustness across individual subject variations.

Each protocol folder includes `.npy`, `.pkl`, and `.tfrecord` files, ready for use with the provided ST-GCN models. All CWBG protocol based implementation can be reproduced using these data.

---

##  Data Pre-processing

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

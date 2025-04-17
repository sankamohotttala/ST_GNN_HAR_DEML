from matplotlib import pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns

"""
This script generates and saves confusion matrices (normalized and default) as heatmaps 
for a classification task. It reads a CSV file containing real and predicted labels, 
computes the confusion matrix, normalizes it, and visualizes the results using seaborn.
Modules:
    - matplotlib.pyplot: For plotting the heatmaps.
    - numpy: For numerical operations.
    - os: For file path manipulations.
    - pandas: For handling CSV data.
    - seaborn: For creating heatmaps.
Constants:
    - trial: Specifies the type of trial (e.g., 'shared', 'similar', 'dissimilar', 'full').
    - file_path: Path to the input CSV file containing real and predicted labels.
    - out_path: Directory where the output heatmaps will be saved.
    - __: Dictionary mapping trial types to the number of classes (cn).
    - cn: Number of classes for the specified trial type.
    - classList: List of class labels.
Functions:
    - None explicitly defined. The script executes sequentially.
Workflow:
    1. Reads the input CSV file specified by `file_path`.
    2. Extracts real and predicted labels from the CSV file.
    3. Computes the confusion matrix (`conf_tmp`) and the count of real labels (`conf_num_label`).
    4. Normalizes the confusion matrix (`conf_norm_default`) to represent percentages.
    5. Generates and saves two heatmaps:
        - Normalized confusion matrix.
        - Default confusion matrix (raw counts).
    6. Saves the heatmaps as images in the specified `out_path`.
Output:
    - Two heatmap images saved in the `out_path` directory:
        1. Normalized confusion matrix: `<trial>_conf_norm.jpg`
        2. Default confusion matrix: `<trial>_conf_default.jpg`
"""

trial='shared' #similar, dissimilar, full,shared
# file_path=r"F:\Codes\joint attention\2022 - Journal\New TF and NTU-44  implementations\ST_GNN_results\results\full_epoch30.csv"
# file_path=r"F:\Codes\joint attention\2022 - Journal\New TF and NTU-44  implementations\ST_GNN_results\results\full_ensemble.csv" 
# file_path=r"F:\Codes\joint attention\2022 - Journal\New TF and NTU-44  implementations\ST_GNN_results\sandbox\full_epoch30.csv"

# file_path=r"F:\Codes\joint attention\2022 - Journal\New TF and NTU-44  implementations\ST_GNN_results\ST-GCN\Sh\data_30.csv"
file_path=r"F:\Codes\joint attention\2022 - Journal\New TF and NTU-44  implementations\CWBG-Sh joint only\stgat_shared_epoch30.csv"
# out_path=r'F:\Codes\joint attention\2022 - Journal\New TF and NTU-44  implementations\ST_GNN_results\ST-GCN\Sh'
out_path=r'F:\Codes\joint attention\2022 - Journal\New TF and NTU-44  implementations\CWBG-Sh joint only'

__={'full':15 ,'similar':10, 'dissimilar':10, 'shared':5}
cn=__[trial]

classList=list(range(1,cn+1))
#get only the directories as a list

# file_path_list=[os.path.join(dir_path,dir_path_new,'save_values_csv','data_30.csv') for dir_path_new in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path,dir_path_new))]

#create a single df with sample label and predicted label
#create a single df from a list of dataframes by concatenating them
# df=pd.concat([pd.read_csv(path) for path in file_path_list])

df = pd.read_csv(file_path)
df_for_conf=df.iloc[:,[1,3]]
 
conf_tmp=np.zeros((cn,cn))
conf_num_label=np.zeros((cn,1)) 
for i in range(df_for_conf.shape[0]):
    
    real_label=df_for_conf.iloc[i,0]
    predict_label=df_for_conf.iloc[i,1]

    conf_tmp[real_label,predict_label]+=1
    conf_num_label[real_label]+=1
    pass
a=7

#normalize the confusion matrix
conf_norm_default=np.zeros((cn,cn))
for i in range(cn):
    conf_norm_default[i,:]=(conf_tmp[i,:]/conf_num_label[i])*100
    pass

plt.figure(figsize=(10, 8)) #(60, 48) changed to 10,8
sns.heatmap(conf_norm_default,xticklabels=classList,yticklabels=classList,  annot=True, fmt='.2f')
plt.xlabel('Prediction')
plt.ylabel('Label')
#plt.savefig('images/conf{}.jpg'.format(test_ite))
plt.savefig(os.path.join(out_path,f'{trial}_conf_norm.jpg'))
plt.close()

plt.figure(figsize=(10, 8)) #(60, 48) changed to 10,8
sns.heatmap(conf_tmp,xticklabels=classList,yticklabels=classList,  annot=True, fmt='.2f')
plt.xlabel('Prediction')
plt.ylabel('Label')
#plt.savefig('images/conf{}.jpg'.format(test_ite))
plt.savefig(os.path.join(out_path,f'{trial}_conf_default.jpg'))
plt.close()

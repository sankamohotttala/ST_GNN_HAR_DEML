import numpy as np
import os
import sys

epcohs_to_use=29 #max 29 if 30, min 0

#go through all folders and get the sub folders
path=r'F:\Codes\joint attention\2022 - Journal\zST-GCN_CWBG_LOOP\results_age_wise\combined_age_mix_f'
base_name=os.path.basename(path)

file_path_list=[]
for root, dirs, files in os.walk(path):
    root_base = os.path.basename(os.path.dirname(root)) #used to only go one depth down
    if 'log' in dirs and root_base==base_name: #used to only go one depth down and only into log folder
        # print(dirs)
        file_path_list.append(os.path.join(root, 'log','accuracies.txt'))
# print(file_path_list)

#get the accuracy from the file
acc_list=[]
for experi in file_path_list:
    with open(experi) as f:
        lines = f.readlines()
        lines = [x.strip() for x in lines]
        check=f'epoch {epcohs_to_use}:'
        for line in lines:
            if check in line:
                # print(line)
                acc=line.split(check)[1]
                acc_list.append(float(acc))
        # print(lines)

#calculate the mean and std
acc_list=np.array(acc_list)
print(f'epoch {epcohs_to_use}:')
print(f'mean: {np.mean(acc_list)}')
print(f'std: {np.std(acc_list)}')
print(f'number of experiments/files: {len(file_path_list)}')

#write the results to a file
with open(os.path.join(path,'repeated_validation_results.txt'),'w') as f:
    f.write(f'epoch {epcohs_to_use}:\n')
    f.write(f'mean: {np.mean(acc_list)}\n')
    f.write(f'std: {np.std(acc_list)}\n')
    f.write(f'number of experiments/files: {len(file_path_list)}\n')

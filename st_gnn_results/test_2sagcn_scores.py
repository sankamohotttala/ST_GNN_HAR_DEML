import sys
import os
import pickle
import numpy as np
from torch import nn
import torch
import pandas as pd 

def selected_classes(trial):
    if trial=='full':
        selected_classes=list(range(1,16))#start from 1 and go to 15
    elif trial=='similar':
        selected_classes=list(range(1,11))
    elif trial=='dissimilar':
        selected_classes=[2,3,4,6,8,9,10,13,14,15]
    elif trial=='shared':
        selected_classes=[9,10,4,14,2]
    else:
        ValueError('trial not found')
    return selected_classes

trial='shared' #similar, dissimilar, full,shared
# out_path=r'F:\Codes\joint attention\2022 - Journal\New TF and NTU-44  implementations\ST_GNN_results\MS-AAGCN\Sh'
out_path =r'F:\Codes\joint attention\2022 - Journal\New TF and NTU-44  implementations\CWBG-Sh joint only'

# pickle_path=r"F:\Codes\joint attention\2022 - Journal\New TF and NTU-44  implementations\ST_GNN_results\epoch30_test_score.pkl"
# pickle_path=r"F:\Codes\joint attention\2022 - Journal\New TF and NTU-44  implementations\ST_GNN_results\ensemble_scores.pkl"

# pickle_path=r"F:\Codes\joint attention\2022 - Journal\New TF and NTU-44  implementations\ST_GNN_results\sandbox\epoch30_test_score.pkl"
# pickle_path=r"F:\Codes\joint attention\2022 - Journal\New TF and NTU-44  implementations\ST_GNN_results\MS-AAGCN\Sh\ensemble_epoch30_scores.pkl"

pickle_path=r"F:\Codes\joint attention\2022 - Journal\New TF and NTU-44  implementations\CWBG-Sh joint only\epoch30_test_score_msaagcn.pkl"


__={'full':15 ,'similar':10, 'dissimilar':10, 'shared':5}
num_classes=__[trial]
class_labels=selected_classes(trial)   #start from 1 NOT 0 
# sys.path.append(r'C:\Users\USER\Downloads')

with open(pickle_path,'rb') as f:
    data=pickle.load(f)


for key,val in data.items():
    print(key,val)
    print(type(key),type(val))
    label=int(key.split('.')[0].split('A')[1])
    print(label)
    break

#check accuracy of the results
tmp_acc=0
sample_num=0
for key,val in data.items():
    label=int(key.split('.')[0].split('A')[1]) #starts from 1 -N
    # label_index=label-1 #starts from 0 - N-1
    class_id=class_labels.index(label) #from 0 to N-1 (N=5,10,15)
    if class_id==np.argmax(val):
        tmp_acc+=1

    sample_num+=1

acc=(tmp_acc/sample_num)*100
print('top-1 accuracy: ',acc)#we get the same result as in 2s-agcn code so values are correctly stored

    
#create and save confusion matrix(removed - reuse other codes with created csv file)
#create csv file
labels=['uniqueID',	'real_label', 'real_probability',	'predicted_label',	'predicted_probability']
classlabels=[str(i)+'_class_probability' for i in range(num_classes)]
final_colunms=labels+classlabels

tmp_list=[]
for key,val in data.items():
    label=int(key.split('.')[0].split('A')[1]) #starts from 1 -15
    class_id=class_labels.index(label) #from 0 to N-1 (N=5,10,15)
    val_new=val.reshape(1,-1) #reshape to (1,N) rather than (N,)
    val_new_torch=torch.from_numpy(val_new)

    softmax_array=nn.Softmax(dim=1)(val_new_torch).numpy()
    tmp=[key.split('.')[0]]
    tmp.append(class_id) #real label
    tmp.append(softmax_array[0,class_id]) #real probability

    predict_label=np.argmax(softmax_array[0])
    tmp.append(predict_label) #predicted label
    tmp.append(softmax_array[0,predict_label]) #predicted probability

    #numpy array to list
    tmp.extend(softmax_array[0].tolist())

    tmp_list.append(tmp)

#list of lists to numpy array
tmp_array=np.array(tmp_list)
a=1

#numpy array to dataframe with column names from a list
df=pd.DataFrame(tmp_array,columns=final_colunms)

#save dataframe to csv without the indexes
input_epoch=os.path.basename(pickle_path).split('_')[0]
df.to_csv(os.path.join(out_path,trial+f'_{input_epoch}.csv'),index=False)








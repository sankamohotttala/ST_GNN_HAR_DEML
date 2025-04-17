#code is used to get all the results from each epoch and save them as a combined score.pkl and acc to a text file
import argparse
import pickle

import numpy as np
from tqdm import tqdm
import os

parser = argparse.ArgumentParser()
# parser.add_argument('--datasets', default='ntu/xsub', choices={'kinetics', 'ntu/xsub', 'ntu/xview'},
#                     help='the work folder for storing results')
parser.add_argument('--alpha', default=1, help='weighted summation')
arg = parser.parse_args()

# dataset = arg.datasets
# label = open('./data/' + dataset + '/val_label.pkl', 'rb')
# label = np.array(pickle.load(label))
# r1 = open('./work_dir/' + dataset + '/agcn_test_joint/epoch1_test_score.pkl', 'rb')
# r1 = list(pickle.load(r1).items())
# r2 = open('./work_dir/' + dataset + '/agcn_test_bone/epoch1_test_score.pkl', 'rb')
# r2 = list(pickle.load(r2).items())

# dataset = r'F:\Codes\joint attention\2022 - Journal\z2s-AGCN\2s-AGCN\checkpoints'
dataset_1=r"F:\Codes\joint attention\2022 - Journal\New TF and NTU-44  implementations\ST_GNN_results\2s-AGCN\Sh\ensemble_scores.pkl" # 2sagcn
dataset_2=r"F:\Codes\joint attention\2022 - Journal\New TF and NTU-44  implementations\ST_GNN_results\MS-AAGCN\Sh\ensemble_epoch30_scores.pkl" # msaagcn
dataset_3=r"F:\Codes\joint attention\2022 - Journal\New TF and NTU-44  implementations\ST_GNN_results\RA-GCN\Sh\epoch40_test_score.pkl" #ragcn
dataset_4=r"F:\Codes\joint attention\2022 - Journal\New TF and NTU-44  implementations\ST_GNN_results\STGAT\Sh\ensemble_epoch30_scores.pkl" #stgat

new_ensemble_score_path=r'F:\Codes\joint attention\2022 - Journal\New TF and NTU-44  implementations\ST_GNN_results\z_ensemble_STGNN'

# pickle_val_path=r"E:\SLIIT RA\Weekly stuff 2 - New approach\Jounal Paper\2s-AGCN\2s-AGCN\2s-AGCN\data\output_cwbg\xsub\val_label.pkl"
pickle_val_path=r"F:\Codes\joint attention\2022 - Journal\New TF and NTU-44  implementations\ST_GNN_results\z_ensemble_STGNN\New folder\val_label.pkl"


# same_epoch=29
for same_epoch in range(1,31):


    # r1_epoch=26
    # r2_epoch=26
    # r3_epoch=26
    # r4_epoch=26



    label = open( pickle_val_path, 'rb')
    label = np.array(pickle.load(label)) #real labels ->   0 - N-1

    r1 = open( dataset_1 , 'rb')
    r1 = list(pickle.load(r1).items())

    r2 = open(dataset_2 , 'rb')
    r2 = list(pickle.load(r2).items())

    r3 = open( dataset_3 , 'rb')
    r3 = list(pickle.load(r3).items())

    r4 = open(dataset_4 , 'rb')
    r4 = list(pickle.load(r4).items())

    right_num = total_num = right_num_5 = 0

    ensemble_scores_dic={name:None for name in label[0]}
    for i in tqdm(range(len(label[0]))):
        _name, l = label[:, i] # l is str type and class index
        _, r11 = r1[i] #r11 is softmax probs from stream 1
        _, r22 = r2[i] #r22 is softmax probs from stream 2
        _, r33 = r3[i] #r11 is softmax probs from stream 1
        _, r44 = r4[i] #r22 is softmax probs from stream 2
        r = r11 + r22  + r33 + r44 # r is the sum of softmax probs from both streams

        #for new ensemble score file - for use in csv creation
        assert label[0,i]==_name
        ensemble_scores_dic[_name]=r
        
        rank_5 = r.argsort()[-5:] #indexs of sorted probs wher ascending order is done 
        right_num_5 += int(int(l) in rank_5)
        r = np.argmax(r)
        right_num += int(r == int(l))
        total_num += 1
    acc = right_num / total_num
    acc5 = right_num_5 / total_num
    print(acc, acc5)

    #save the ensemble scores as a pickle file
    name=f'ensemble_epoch{same_epoch}_scoresSh.pkl'
    with open(os.path.join(new_ensemble_score_path,name), 'wb') as f:
        pickle.dump(ensemble_scores_dic, f)

    #save acc to a text file
    name_text='ensemble_accSh.txt'
    with open(os.path.join(new_ensemble_score_path,name_text),'a+') as f:
        f.write('epoch: '+str(same_epoch)+' acc: '+str(acc)+' top-5 acc: '+str(acc5)+'\n')
    break 


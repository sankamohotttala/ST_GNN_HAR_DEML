import argparse
import pickle

import numpy as np
from tqdm import tqdm

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
dataset_1=r'E:\SLIIT RA\Weekly stuff 2 - New approach\Jounal Paper\2s-AGCN\2s-AGCN\2s-AGCN\results-aagcn\cwbg_full_1_new_lr' # stream - J-M
dataset_2=r'E:\SLIIT RA\Weekly stuff 2 - New approach\Jounal Paper\2s-AGCN\2s-AGCN\2s-AGCN\results-aagcn\cwbg_full_2' # stream - B-M
dataset_3=r'E:\SLIIT RA\Weekly stuff 2 - New approach\Jounal Paper\2s-AGCN\2s-AGCN\2s-AGCN\results-aagcn\cwbg_full_B_1' #stream - B
dataset_4=r'E:\SLIIT RA\Weekly stuff 2 - New approach\Jounal Paper\2s-AGCN\2s-AGCN\2s-AGCN\results-aagcn\cwbg_full_J_1' #stream - J

r1_epoch=26
r2_epoch=26
r3_epoch=26
r4_epoch=26

pickle_val_path=r"E:\SLIIT RA\Weekly stuff 2 - New approach\Jounal Paper\2s-AGCN\2s-AGCN\2s-AGCN\data\output_cwbg\xsub\val_label.pkl"

label = open( pickle_val_path, 'rb')
label = np.array(pickle.load(label)) #real labels ->   0 - N-1

r1 = open( dataset_1 + f'/epoch{r1_epoch}_test_score.pkl', 'rb')
r1 = list(pickle.load(r1).items())

r2 = open(dataset_2 + f'/epoch{r2_epoch}_test_score.pkl', 'rb')
r2 = list(pickle.load(r2).items())

r3 = open( dataset_3 + f'/epoch{r3_epoch}_test_score.pkl', 'rb')
r3 = list(pickle.load(r3).items())

r4 = open(dataset_4 + f'/epoch{r4_epoch}_test_score.pkl', 'rb')
r4 = list(pickle.load(r4).items())

right_num = total_num = right_num_5 = 0
for i in tqdm(range(len(label[0]))):
    _, l = label[:, i] # l is str type and class index
    _, r11 = r1[i] #r11 is softmax probs from stream 1
    _, r22 = r2[i] #r22 is softmax probs from stream 2
    _, r33 = r3[i] #r11 is softmax probs from stream 1
    _, r44 = r4[i] #r22 is softmax probs from stream 2
    r = r11 + r22  + r33 + r44 # r is the sum of softmax probs from both streams
    
    rank_5 = r.argsort()[-5:] #indexs of sorted probs wher ascending order is done 
    right_num_5 += int(int(l) in rank_5)
    r = np.argmax(r)
    right_num += int(r == int(l))
    total_num += 1
acc = right_num / total_num
acc5 = right_num_5 / total_num
print(acc, acc5)


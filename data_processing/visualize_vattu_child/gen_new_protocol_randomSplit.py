#this contains the code for random split in a balanced way
import argparse
import pickle
from tqdm import tqdm
import sys
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

sys.path.extend(['../'])
from pre_process import pre_normalization

class_set='shared' # full, similar, dissimilar, shared
subject_set='all_subjects' # boys, 3_4, 4_5, 3_5, all_subjects

#train subject set selection
subjects_protocols={'all_subjects':list(range(1,31))}
training_subjects=subjects_protocols[subject_set]

#train class set selection
class_protocols={'full':list(range(1,16)),'similar':list(range(1,11)),
                        'dissimilar':[2,3,4,6,8,9,10,13,14,15],'shared':[9,10,4,14,2]}
action_class_list=class_protocols[class_set]


# training_subjects = list(range(1,22))  # 1--21 are used for training

training_cameras = [2, 3] #this is of no use
max_body_true = 1
max_body_kinect = 4
num_joint = 20
max_frame = 300 #this is an issue
# action_class_list=[9,10,4,14,2] #shared 5 classes between NTU and CFBG
# action_class_list=[14,10,5,8,3] #shared 5 classes between CFBG and K-G -> throw ball,jump,draw a circle,fly like a bird,climb ladder
# action_class_list=list()

trainNum=0
testNum=0
action_name=None

import numpy as np
import os
class Param:
    pass
paramss=Param()
paramss.action=None

def classDictionary(filename):
    tmpDict={}
    for classNum in range(1,121):
        tmpDict[classNum]=''
    with open(filename) as file:
        lines = file.readlines()
        lines = [line.rstrip() for line in lines]
        for line in lines:
             
            tmpNum=int(line[line.find('A')+1 :line.find(':') ])
            tmpName=line[line.find(' ')+1 : ]
            tmpDict[tmpNum]+=tmpName
    return tmpDict

def read_skeleton_filter(file): #removed paraamss from parameters
    with open(file, 'r') as f:
        skeleton_sequence = {}
        skeleton_sequence['numFrame'] = int(f.readline())

        #paramss.fram_list[skeleton_sequence['numFrame']-1]+=1

        skeleton_sequence['frameInfo'] = []
        # num_body = 0
        for t in range(skeleton_sequence['numFrame']):
            frame_info = {}
            frame_info['numBody'] = int(f.readline())
            frame_info['bodyInfo'] = []

            for m in range(frame_info['numBody']):
                body_info = {}
                body_info_key = [
                    'bodyID', 'clipedEdges', 'handLeftConfidence',
                    'handLeftState', 'handRightConfidence', 'handRightState',
                    'isResticted', 'leanX', 'leanY', 'trackingState'
                ]
                body_info = {
                    k: float(v)
                    for k, v in zip(body_info_key, f.readline().split())
                }
                body_info['numJoint'] = int(f.readline())
                body_info['jointInfo'] = []
                for v in range(body_info['numJoint']):
                    joint_info_key = [
                        'x', 'y', 'z', 'depthX', 'depthY', 'colorX', 'colorY',
                        'orientationW', 'orientationX', 'orientationY',
                        'orientationZ', 'trackingState'
                    ]
                    joint_info = {
                        k: float(v)
                        for k, v in zip(joint_info_key, f.readline().split())
                    }
                    body_info['jointInfo'].append(joint_info)
                frame_info['bodyInfo'].append(body_info)
            skeleton_sequence['frameInfo'].append(frame_info)

    return skeleton_sequence


def get_nonzero_std(s):  # tvc
    index = s.sum(-1).sum(-1) != 0  # select valid frames
    s = s[index]
    if len(s) != 0:
        s = s[:, :, 0].std() + s[:, :, 1].std() + s[:, :, 2].std()  # three channels
    else:
        s = 0
    return s


def read_xyz(file, max_body=4, num_joint=25):  # removed paramss 
    seq_info = read_skeleton_filter(file)
    data = np.zeros((max_body, seq_info['numFrame'], num_joint, 3))
    for n, f in enumerate(seq_info['frameInfo']):
        for m, b in enumerate(f['bodyInfo']):
            for j, v in enumerate(b['jointInfo']):
                if m < max_body and j < num_joint:
                    data[m, n, j, :] = [v['x'], v['y'], v['z']]
                else:
                    pass
    
    #trimming the video has to be done here as well
    # select two max energy body
    energy = np.array([get_nonzero_std(x) for x in data])
    index = energy.argsort()[::-1][0:max_body_true]
    data = data[index]
    data = data.transpose(3, 1, 2, 0)
    return data


def gendata(data_path, out_path, ignored_sample_path=None, benchmark='xview', part='eval'):
    # if ignored_sample_path != None:
    #     with open(ignored_sample_path, 'r') as f:
    #         ignored_samples = [
    #             line.strip() + '.skeleton' for line in f.readlines()
    #         ]
    # else:
    #     ignored_samples = []
    
    sample_name = []
    sample_label = []

    subject_id_list_=[]
    action_classes_=[]
    for filename in os.listdir(data_path):
        # if filename in ignored_samples:
        #     continue
        action_class = int(
            filename[filename.find('A') + 1:filename.find('A') + 4])
        subject_id = int(
            filename[filename.find('P') + 1:filename.find('P') + 4])
        camera_id = int(
            filename[filename.find('C') + 1:filename.find('C') + 4])

        subject_id_list_.append(subject_id)
        action_classes_.append(action_class)
        
        if benchmark == 'xview':
            istraining = (camera_id in training_cameras)
        elif benchmark == 'xsub':
            istraining = (subject_id in training_subjects)
        else:
            raise ValueError()

        if part == 'train':
            issample = istraining
        elif part == 'val':
            issample = not (istraining)
        else:
            raise ValueError()
        assert issample == True #since by definition we are using all subjects
        if issample and (action_class in action_class_list):
            sample_name.append(filename)
            index=action_class_list.index(action_class)
            sample_label.append(index)
        # sample_name.append(filename)
        # sample_label.append(action_class - 1)
    

    
    fileActionNames=r'F:\Codes\joint attention\2022\visualize_vattu_child\ListClassNames.txt'
    action_name=classDictionary(fileActionNames)

    sample_name_train,sample_name_test,sample_label_train,sample_label_test=train_test_split(sample_name,
                                                             sample_label, test_size=0.30, random_state=1, stratify=sample_label)

    split={'train':(sample_name_train,sample_label_train),'val':(sample_name_test,sample_label_test)}

    for part in ['train','val']:
        #add for loop for train and val splits
        sample_name=split[part][0]
        sample_label=split[part][1]

        with open('{}/{}_label.pkl'.format(out_path,part), 'wb') as f:
            pickle.dump((sample_name, list(sample_label)), f)

        fp = np.zeros((len(sample_label), 3, max_frame, num_joint, max_body_true), dtype=np.float32)

            
        for i, s in enumerate(tqdm(sample_name)):
            data = read_xyz(os.path.join(data_path, s), max_body=max_body_kinect, num_joint=num_joint)
            fp[i, :, 0:data.shape[1], :, :] = data
        a_o=2+2
        fp = pre_normalization(fp,out_path,action_name)
        np.save('{}/{}_data_joint.npy'.format(out_path,part), fp)


if __name__ == '__main__':
    
    #later use to remove the files not needed for the testing from processing
    ignored_sample_path=r'F:\Data Sets\Skeleton\3D\Adult\NTU dataset\_____with_missing_skeletons.txt'
    data_path_final=r'F:\Codes\joint attention\2022\visualize_vattu_child\test_files'
    #data_path_final=r'F:\Codes\joint attention\2022\visualize_vattu_child\tmp'
    output_main_folder=r'F:\Codes\joint attention\2022\visualize_vattu_child\new_protocols\random_split'
    output_folder=os.path.join(output_main_folder,class_set)


    benchmark = ['xsub']
    part = ['train']

    for b in benchmark:
        for p in part:
            out_path = os.path.join(output_folder,b)
            if not os.path.exists(out_path):
                os.makedirs(out_path)
            #print(b, p)
            gendata(data_path_final,
                out_path,
                ignored_sample_path,benchmark=b,part=p)

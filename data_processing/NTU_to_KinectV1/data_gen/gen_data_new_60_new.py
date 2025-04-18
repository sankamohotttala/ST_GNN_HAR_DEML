#this is for the new ntu-60 classes based implementations; 
#redone for journal paper implementations
import argparse
import pickle
from tqdm import tqdm
import sys
from matplotlib import pyplot as plt

sys.path.extend(['../'])
from pre_process import pre_normalization

# classes_list='worst'
training_subjects = [
    1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 31, 34, 35, 38
]

# training_subjects=[1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 31, 34, 35,
#     38, 45, 46, 47, 49, 50, 52, 53, 54, 55, 56, 57, 58, 59, 70, 74, 78,
#     80, 81, 82, 83, 84, 85, 86, 89, 91, 92, 93, 94, 95, 97, 98, 100, 103]

training_cameras = [2, 3]
max_body_true = 1 # changed from 2
max_body_kinect = 4
num_joint = 21 # changed from 25
max_frame = 300

# if classes_list=='best':#taken as the best performing 44 classes of ntu-120
#     action_classes_used=[27, 43, 26, 55, 8, 97, 9, 42, 113, 60, 14, 101, 112, 58, 24, 120, 6, 114, 98, 18, 36, 19, 52, 49, 59, 38, 
#                      57, 99, 115, 80, 100, 32, 64, 23, 15, 56, 1, 22, 53, 39, 35, 45, 33, 66] 
# elif classes_list=='worst': #taken as the worst performing 44 classes of ntu-120
#     action_classes_used=[77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101,
#              102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120]
# else:
#     raise ValueError('classes_list should be either best or worst') 

action_classes_used=list(range(1,61))

#interaction_classes_used=[59,60]#and 2 interaction classes

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
    data = np.zeros((max_body, seq_info['numFrame'], num_joint, 3)) #shape of (4,#frame,20,3)
    for n, f in enumerate(seq_info['frameInfo']):
        for m, b in enumerate(f['bodyInfo']):
            for j, v in enumerate(b['jointInfo']):
                if m < max_body and j < num_joint:
                    data[m, n, j, :] = [v['x'], v['y'], v['z']]
                else:
                    pass
    
    # replace the #3 joint
    data=np.delete(data,2,2)
    # select two max energy body
    energy = np.array([get_nonzero_std(x) for x in data])
    index = energy.argsort()[::-1][0:max_body_true]
    data = data[index]

    data = data.transpose(3, 1, 2, 0)
    return data


def gendata(data_path, out_path,extra_skeleton_path_60=None,extra_skeleton_path_120=None, ignored_sample_path=None, benchmark='xview', part='eval'):
    if ignored_sample_path != None:
        with open(ignored_sample_path, 'r') as f:
            ignored_samples = [
                line.strip() + '.skeleton' for line in f.readlines()
            ]
    else:
        ignored_samples = []

    if extra_skeleton_path_60 !=None:
        with open(extra_skeleton_path_60, 'r') as f:
            extra_skeleton_samples_1 = [
                line.strip()  for line in f.readlines()
            ]      
    else:
        extra_skeleton_samples_1=[]

    if extra_skeleton_path_120 !=None:
        with open(extra_skeleton_path_120, 'r') as f:
            extra_skeleton_samples_2 = [
                line.strip()  for line in f.readlines()
            ]
    else:
        extra_skeleton_samples_2=[]

    extra_skeleton_samples=extra_skeleton_samples_1+extra_skeleton_samples_2


    sample_name = []
    sample_label = []
    for filename in os.listdir(data_path):
        if filename in ignored_samples:
            continue
        if filename in extra_skeleton_samples:
            continue

        action_class = int(
            filename[filename.find('A') + 1:filename.find('A') + 4])
        subject_id = int(
            filename[filename.find('P') + 1:filename.find('P') + 4])
        camera_id = int(
            filename[filename.find('C') + 1:filename.find('C') + 4])

        
        if benchmark == 'xview':
            istraining = (camera_id in training_cameras)
        elif benchmark == 'xsub':
            if action_class<61: # 1 -60 only
                istraining = (subject_id in training_subjects)
            else:
                istraining = (subject_id not in training_subjects)
        else:
            raise ValueError()

        if part == 'train':
            issample = istraining
        elif part == 'val':
            issample = not (istraining)
        else:
            raise ValueError()

        if issample and (action_class in action_classes_used):
            sample_name.append(filename)
            #sample_label.append(action_class - 1)   #1->0 .....,..,.... 49->48
            index=action_classes_used.index(action_class)
            sample_label.append(index)

    
    action_name=classDictionary(fileActionNames)
    with open('{}/{}_label.pkl'.format(out_path, part), 'wb') as f:
        pickle.dump((sample_name, list(sample_label)), f)

    fp = np.zeros((len(sample_label), 3, max_frame, 20, max_body_true), dtype=np.float32) #this contain at this point a 300/900 frames

        
    for i, s in enumerate(tqdm(sample_name)):
        data = read_xyz(os.path.join(data_path, s), max_body=max_body_kinect, num_joint=num_joint)
        fp[i, :, 0:data.shape[1], :, :] = data
    a_o=2+2
    fp = pre_normalization(fp,out_path,action_name,part)# after this resultant fp contain the downsamples one with frames 300/100
    np.save('{}/{}_data_joint.npy'.format(out_path, part), fp)


if __name__ == '__main__':

    # fileActionNames=r"F:\Codes\joint attention\2022\NTU_to_KinectV1\data_new_44\top_44_classes.txt"
    fileActionNames=r"F:\Codes\joint attention\2022\NTU_to_KinectV1\__NTU_full_class_name_list.txt"

    ignored_sample_path=r'F:\Data Sets\Skeleton\3D\Adult\NTU dataset\processed skeleton data\NTU-120-skeleton-full\full_dataset_missing_skeletons.txt'
    ignore_extra_skeleton_path_60=r'F:\Data Sets\Skeleton\3D\Adult\NTU dataset\processed skeleton data\NTU-120-skeleton-full\ntu_60_extra_skeleton.txt'
    ignore_extra_skeleton_path_120=r'F:\Data Sets\Skeleton\3D\Adult\NTU dataset\processed skeleton data\NTU-120-skeleton-full\ntu_120_extra_skeleton.txt'
    
    data_path_final=r'F:\Data Sets\Skeleton\3D\Adult\NTU dataset\processed skeleton data\Processed skeleton files\nturgbd_skeletons_s001_to_s017\nturgb+d_skeletons'
    # data_path_final=r'F:\Data Sets\Skeleton\3D\Adult\NTU dataset\processed skeleton data\NTU-120-skeleton-full\Data-tmp'
    output_folder=r'F:\Codes\joint attention\2022\NTU_to_KinectV1\data_new_60_new'

    benchmark = ['xsub']
    part = ['train', 'val']

    for b in benchmark:
        for p in part:
            out_path = os.path.join(output_folder, b)
            if not os.path.exists(out_path):
                os.makedirs(out_path)
            print(b, p)
            gendata(data_path_final,
                out_path,
                ignore_extra_skeleton_path_60,ignore_extra_skeleton_path_120,
                ignored_sample_path,
                benchmark=b,
                part=p)

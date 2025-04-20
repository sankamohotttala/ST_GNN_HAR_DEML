import argparse
import pickle
from tqdm import tqdm
import sys
from matplotlib import pyplot as plt

sys.path.extend(['../'])
from pre_process import pre_normalization

training_subjects = list(range(1,22))  # 1--21 are used for training
# training_subjects=[1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 31, 34,
#     35,38, 45, 46, 47, 49, 50, 52, 53, 54, 55, 56, 57, 58, 59, 70, 74, 78, 80, 81, 82,
#     83, 84, 85, 86, 89, 91, 92, 93, 94, 95, 97, 98, 100, 103]
training_cameras = [2, 3] #this is of no use
max_body_true = 1
max_body_kinect = 4
num_joint = 20
max_frame = 300 #this is an issue

trainNum=0
testNum=0
action_name=None

import numpy as np
import os
class Param:
    pass
paramss=Param()
paramss.action=None
'''paramss.k=None
paramss.fram_list=[i for i in range(300) ]'''
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
    '''#    previous=data[0,0,:,:]
    diff_sum=0*data[0,0,:,:]
    previous=0*data[0,0,:,:]
    for i,_ in enumerate(seq_info['frameInfo']):
        new=data[0,i,:,:]
        diff=new-previous
        diff_new=np.absolute(diff)
        previous=new
        diff_sum+=diff_new
        paramss.k=diff_sum'''
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
    '''camera_list=[0 for _ in range(3)]
    people_list=[0 for _ in range(40)]
    action_class_list=[0 for _ in range(60)]'''
    for filename in os.listdir(data_path):
        # if filename in ignored_samples:
        #     continue
        action_class = int(
            filename[filename.find('A') + 1:filename.find('A') + 4])
        subject_id = int(
            filename[filename.find('P') + 1:filename.find('P') + 4])
        camera_id = int(
            filename[filename.find('C') + 1:filename.find('C') + 4])
        
        '''camera_list[camera_id-1]+=1
        people_list[subject_id-1]+=1
        action_class_list[action_class-1]+=1'''
        #paramss.action=action_class


        
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

        if issample and (action_class<11):
            sample_name.append(filename)
            sample_label.append(action_class - 1)
        # sample_name.append(filename)
        # sample_label.append(action_class - 1)

 
    fileActionNames=r'F:\Codes\joint attention\2022\visualize_vattu_child\ListClassNames.txt'
    action_name=classDictionary(fileActionNames)
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
    output_folder=r'F:\Codes\joint attention\2022\visualize_vattu_child\save_directory_2'


    benchmark = ['xsub']
    part = ['train', 'val']

    for b in benchmark:
        for p in part:
            out_path = os.path.join(output_folder,b)
            if not os.path.exists(out_path):
                os.makedirs(out_path)
            #print(b, p)
            gendata(data_path_final,
                out_path,
                ignored_sample_path,benchmark=b,part=p)

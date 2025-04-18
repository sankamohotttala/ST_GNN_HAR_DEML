import argparse
import pickle
from tqdm import tqdm
import sys
from matplotlib import pyplot as plt

sys.path.extend(['../'])
from preprocess import pre_normalization

training_subjects = [
    1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 31, 34, 35, 38
]
training_cameras = [2, 3]
max_body_true = 1 # changed from 2
max_body_kinect = 4
num_joint = 21 # changed from 25
max_frame = 300

import numpy as np
import os
class Param:    
    pass
paramss=Param()
paramss.action=None
'''paramss.k=None
paramss.fram_list=[i for i in range(300) ]'''

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


def gendata(paramss,data_path, out_path, ignored_sample_path=None, benchmark='xview', part='eval'):
    if ignored_sample_path != None:
        with open(ignored_sample_path, 'r') as f:
            ignored_samples = [
                line.strip() + '.skeleton' for line in f.readlines()
            ]
    else:
        ignored_samples = []
    sample_name = []
    sample_label = []
    '''camera_list=[0 for _ in range(3)]
    people_list=[0 for _ in range(40)]
    action_class_list=[0 for _ in range(60)]'''
    for filename in os.listdir(data_path):
        if filename in ignored_samples:
            continue
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

        if issample:
            sample_name.append(filename)
            sample_label.append(action_class - 1)

    '''bins_cam = np.arange(0, len(camera_list), 1) # fixed bin size
    cam_name=['cam 1','cam 2','cam 3']
    camera_dic=dict(zip(cam_name,camera_list))
    people_name=['P'+str(i) for i in range(40)]
    people_dic=dict(zip(people_name,people_list))
    action_name=['a'+str(i) for i in range(60)]
    action_dic=dict(zip(action_name,action_class_list))
    plt.bar(*zip(*camera_dic.items()))
    #plt.show()
    
    #plt.hist(3,camera_list)
    plt.title('Video Distribution between cameras')
    plt.xlabel('Camera ID')
    plt.ylabel('number of videos')

    plt.show()
    plt.bar(*zip(*people_dic.items()))
    #plt.show()
    
    #plt.hist(3,camera_list)
    plt.title('Video Distribution between Participants')
    plt.xlabel('Participant ID')
    plt.ylabel('number of videos')

    plt.show()
    a=plt.bar(*zip(*action_dic.items()))
    intera=list(range(50,60))
    for i in intera:
        a[i].set_color('r')
    #plt.show()
    
    #plt.hist(3,camera_list)
    plt.title('Video Distribution between Action classes')
    plt.xlabel('Action ID')
    plt.ylabel('number of videos')

    plt.show()'''
    with open('{}/{}_label.pkl'.format(out_path, part), 'wb') as f:
        pickle.dump((sample_name, list(sample_label)), f)

    fp = np.zeros((len(sample_label), 3, max_frame, 20, max_body_true), dtype=np.float32)

        
    for i, s in enumerate(tqdm(sample_name)):
        data = read_xyz(os.path.join(data_path, s), max_body=max_body_kinect, num_joint=num_joint)
        fp[i, :, 0:data.shape[1], :, :] = data
    a_o=2+2
    fp = pre_normalization(fp)
    np.save('{}/{}_data_joint.npy'.format(out_path, part), fp)


if __name__ == '__main__':
    
    ignored_sample_path=r'F:\Codes\joint attention\New folder\tf_changed_vertex\New folder\data\NTU\original\samples_with_missing_skeletons.txt'
    #data_path_final=r'F:\Codes\joint attention\New folder\tf_changed_vertex\New folder\data\NTU\original\nturgb+d_skeletons'
    data_path_final=r'F:\Codes\joint attention\New folder\tf_changed_vertex\New folder\data\NTU\original\nturgb+d_skeletons'
    # data_path_final=r'F:\Codes\joint attention\New folder\tf_changed_vertex\New folder\data\NTU\test'
    output_folder=r'F:\Codes\joint attention\New folder\tf_changed_vertex\New folder\data\NTU\processed_newa'


    parser = argparse.ArgumentParser(description='NTU-RGB-D Data Converter.')
    parser.add_argument('--data_path', default=data_path_final)
    parser.add_argument('--ignored_sample_path',
                        default=ignored_sample_path)
    parser.add_argument('--out_folder', default=output_folder)

    benchmark = ['xsub']
    part = ['train', 'val']
    arg = parser.parse_args()

    for b in benchmark:
        for p in part:
            out_path = os.path.join(arg.out_folder, b)
            if not os.path.exists(out_path):
                os.makedirs(out_path)
            print(b, p)
            gendata(paramss,
                arg.data_path,
                out_path,
                arg.ignored_sample_path,
                benchmark=b,
                part=p)

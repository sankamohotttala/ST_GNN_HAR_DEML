import pickle
import sys
import numpy as np

sys.path.extend(['../'])
from rotation_1 import *
from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle



def pickle_load(filename):
    actionNameList=[]
    infile = open(filename,'rb')
    new_dict = pickle.load(infile)
    infile.close()
    return new_dict[0],new_dict[1]

def first_downsample_method_100frames(s):
    frames_100=list(range(0,300,3))
    T=len(frames_100)
    assert T==100 #check if there is only 100 frames
    s=s[:,:,frames_100,:,:] #resampling/down sampling
    return T,s

def first_downsample_method_300frames(s):
    frames_300=list(range(0,900,3))
    T=len(frames_300)
    assert T==300 #check if there is only 100 frames
    s=s[:,:,frames_300,:,:] #resampling/down sampling
    return T,s

def average_sample(s,kernal_size=3):
    s=s[:,:,:300,:,:]
    N, M, T, V, C=s.shape
    bags=T//kernal_size
    new_s=np.zeros((N,M,bags,V,C), dtype=np.float32) #100 frames
    # final_padded_s=np.zeros((N,M,T,V,C), dtype=np.float32)#300 frames
    for i_s, skeleton in enumerate(tqdm(s)):  # pad
        # if skeleton.sum() == 0:
        #     print(i_s, ' has no skeleton')
        for i_p, person in enumerate(skeleton):
            # if person.sum() == 0:
            #     continue
            # if person[0].sum() == 0:
            #     index = (person.sum(-1).sum(-1) != 0)
            #     tmp = person[index].copy()
            #     person *= 0
            #     person[:len(tmp)] = tmp
            for samplepoint in range(bags):
                # print(samplepoint*kernal_size,samplepoint*kernal_size+3)
                sample=person[samplepoint*kernal_size:samplepoint*kernal_size+3,:,:]
                sampleDown=np.sum(sample,axis=0,keepdims=False)/3
                new_s[i_s,i_p,samplepoint,:,:]=sampleDown
    final_padded_s=np.concatenate((new_s,new_s,new_s),axis=2)
    return T,final_padded_s


def LPF_sample(s,kernal_size=5):
    s=s[:,:,:300,:,:]
    N, M, T, V, C=s.shape
    # bags=T//kernal_size
    new_s=np.zeros((N,M,T,V,C), dtype=np.float32) #100 frames
    # final_padded_s=np.zeros((N,M,T,V,C), dtype=np.float32)#300 frames
    first2=s
    for i_s, skeleton in enumerate(tqdm(s)):  # pad
        # if skeleton.sum() == 0:
        #     print(i_s, ' has no skeleton')
        for i_p, person in enumerate(skeleton):
            first2=person[:2,:,:] # we use the same padding
            last2=person[T-2:,:,:]  # we use the same padding
            for start_frame in range(T-4):
                # print(samplepoint*kernal_size,samplepoint*kernal_size+3)
                # if start_frame+5<=T:
                sample=person[start_frame:start_frame+kernal_size,:,:]
                sampleDown=np.sum(sample,axis=0,keepdims=False)/kernal_size
                new_s[i_s,i_p,start_frame+2,:,:]=sampleDown
            new_s[i_s,i_p,:2,:,:]=first2
            new_s[i_s,i_p,T-2:,:,:]=last2
    # final_new_s=np.concatenate((new_s,new_s,new_s),axis=2)
    return T,new_s

def LPF_sample_with_10FPS(s,kernal_size=5):
    '''
    here we not only use a low pass filter with a sliding window of size 5
    we then take downsample the result by selecting the 3rd sample
    '''
    T,LPF_output=LPF_sample(s,kernal_size)
    final_padded_s=np.concatenate((LPF_output,LPF_output,LPF_output),axis=2)
    T_2,final_s_300=first_downsample_method_300frames(final_padded_s)
    return T_2,final_s_300

                


def visualization_skeleton_all_normal(data,pickleDir,actionDic,part='train'):
    inward_ori_index = [(1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5), (7, 6),
                    (8, 7), (9, 21), (10, 9), (11, 10), (12, 11), (13, 1),
                    (14, 13), (15, 14), (16, 15), (17, 1), (18, 17), (19, 18),
                    (20, 19), (22, 23), (23, 8), (24, 25), (25, 12)]


    # inward_ori_index = [(1, 2), (2, 20), (3,20), (4, 20), (5, 4), (6, 5),
    #                 (7, 6), (8, 20), (9, 8), (10, 9), (11, 10), (12, 1),
    #                 (13, 12), (14, 13), (15, 14), (16, 1), (17, 16), (18, 17),
    #                 (19, 18)]

    N, M, T, V, C = data.shape # N, M, T, V, C


    fileNameList,classNumList=pickle_load(pickleDir+'/{}_label.pkl'.format(part))
    for _ in range(len(fileNameList)):

        index_1=_
        filename=fileNameList[index_1]
        ClassNum=classNumList[index_1]
        ClassName=actionDic[int(ClassNum)+1]#since the classNum starts from 0

        video_0=data[index_1]
        x_all=video_0[0,:,:,0]
        y_all=video_0[0,:,:,1]
        z_all=video_0[0,:,:,2]    
        

        fig = plt.figure()
        ax = plt.axes(projection="3d")

        # ax.set_xlim(np.amin(x_all), np.amax(x_all))
        # ax.set_ylim(np.amin(y_all), np.amax(y_all))
        # ax.set_zlim(np.amin(z_all), np.amax(z_all))

        for frame in range(T):
            for person in range(M):
                skeleton=video_0[person,frame]
                if skeleton.sum()==0:
                    continue
                for bone in inward_ori_index:
                    a,b=[i-1 for i in list(bone)]
                    
                    x_points = [skeleton[a,0],skeleton[b,0]]
                    y_points = [skeleton[a,1],skeleton[b,1]]
                    z_points = [skeleton[a,2],skeleton[b,2]]
                    
                    ax.plot3D(x_points, y_points, z_points,zdir='z', c='black')
                    #ax.plot3D(x_points, z_points,y_points, 'black')    

                x_points = skeleton[:,0]
                y_points = skeleton[:,1]
                z_points = skeleton[:,2]
                ax.scatter3D(x_points, y_points, z_points,zdir='z', c='black')
                for i in range(V):
                    label = str(i)
                    #cordi_label='x:{0:.3f}, y:{1:.3f}, z:{2:.3f}'.format(x_points[i],y_points[i],z_points[i])
                    ax.text(x_points[i], y_points[i], z_points[i], label, zdir=None)
                    #ax.text(x_points[i], y_points[i], z_points[i],cordi_label , zdir=None)
                #ax.scatter3D(x_points, z_points,y_points, c='black')

            ax.text2D(0.05, 0.95, '{}'.format(filename), transform=ax.transAxes)
            ax.text2D(0.95, 0.95, '{}'.format(ClassName), transform=ax.transAxes)
            ax.text2D(0.05, 0.05, 'frameNo: {}'.format(frame), transform=ax.transAxes)
            ax.set_xlim(-1, 1)
            ax.set_ylim(-1, 1)
            ax.set_zlim(-1, 1)
            

            ax.set_xlabel('X axis')
            ax.set_ylabel('Y axis')
            ax.set_zlabel('Z axis')
            



            
            plt.draw()
            plt.pause(.1)   
            ax.cla() # put the breakpoint here

        plt.close()


def visualization_skeleton_all(data,pickleDir,actionDic,part='train'):
    # inward_ori_index = [(1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5), (7, 6),
    #                 (8, 7), (9, 21), (10, 9), (11, 10), (12, 11), (13, 1),
    #                 (14, 13), (15, 14), (16, 15), (17, 1), (18, 17), (19, 18),
    #                 (20, 19), (22, 23), (23, 8), (24, 25), (25, 12)]


    inward_ori_index = [(1, 2), (2, 20), (3,20), (4, 20), (5, 4), (6, 5),
                    (7, 6), (8, 20), (9, 8), (10, 9), (11, 10), (12, 1),
                    (13, 12), (14, 13), (15, 14), (16, 1), (17, 16), (18, 17),
                    (19, 18)]

    N, M, T, V, C = data.shape # N, M, T, V, C

    #index_1=3 #can be changed to select the video
    fileNameList,classNumList=pickle_load(pickleDir+'/{}_label.pkl'.format(part))
    for _ in range(len(fileNameList)):

        index_1=_
        filename=fileNameList[index_1]
        ClassNum=classNumList[index_1]
        ClassName=actionDic[int(ClassNum)+1]#since the classNum starts from 0

        video_0=data[index_1]
        x_all=video_0[0,:,:,0]
        y_all=video_0[0,:,:,1]
        z_all=video_0[0,:,:,2]    
        

        fig = plt.figure()
        ax = plt.axes(projection="3d")

        # ax.set_xlim(np.amin(x_all), np.amax(x_all))
        # ax.set_ylim(np.amin(y_all), np.amax(y_all))
        # ax.set_zlim(np.amin(z_all), np.amax(z_all))

        for frame in range(T):
            for person in range(M):
                skeleton=video_0[person,frame]
                if skeleton.sum()==0:
                    continue
                for bone in inward_ori_index:
                    a,b=[i-1 for i in list(bone)]
                    
                    x_points = [skeleton[a,0],skeleton[b,0]]
                    y_points = [skeleton[a,1],skeleton[b,1]]
                    z_points = [skeleton[a,2],skeleton[b,2]]
                    
                    ax.plot3D(x_points, y_points, z_points,zdir='z', c='black')
                    #ax.plot3D(x_points, z_points,y_points, 'black')    

                x_points = skeleton[:,0]
                y_points = skeleton[:,1]
                z_points = skeleton[:,2]
                ax.scatter3D(x_points, y_points, z_points,zdir='z', c='black')
                for i in range(V):
                    label = str(i)
                    cordi_label='x:{0:.3f}, y:{1:.3f}, z:{2:.3f}'.format(x_points[i],y_points[i],z_points[i])
                    #ax.text(x_points[i], y_points[i], z_points[i], label, zdir=None)       # joint number
                    #ax.text(x_points[i], y_points[i], z_points[i],cordi_label , zdir=None)  # x,y,z values
                #ax.scatter3D(x_points, z_points,y_points, c='black')

            ax.text2D(0.05, 0.95, '{}'.format(filename), transform=ax.transAxes)
            ax.text2D(0.95, 0.95, '{}'.format(ClassName), transform=ax.transAxes)
            ax.text2D(0.05, 0.05, 'frameNo: {}'.format(frame), transform=ax.transAxes)
            ax.set_xlim(-1, 1)
            ax.set_ylim(-1, 1)
            ax.set_zlim(-1, 1)
            

            ax.set_xlabel('X axis')
            ax.set_ylabel('Y axis')
            ax.set_zlabel('Z axis')
            



            if frame==0 or True: # this is used so that we can get the first frame and only first frame : to analyse the skeleton orientation 
                plt.draw()
                plt.pause(.1)   
                ax.cla() # put the breakpoint here
            else:
                #break   #use to just get the first frame of each video
                pass
        plt.close()

def visualization_skeleton_one(data,pickleDir,actionDic,part='train'):
    inward_ori_index = [(1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5), (7, 6),
                    (8, 7), (9, 21), (10, 9), (11, 10), (12, 11), (13, 1),
                    (14, 13), (15, 14), (16, 15), (17, 1), (18, 17), (19, 18),
                    (20, 19), (22, 23), (23, 8), (24, 25), (25, 12)]


    # inward_ori_index = [(1, 2), (2, 20), (3,20), (4, 20), (5, 4), (6, 5),
    #                 (7, 6), (8, 20), (9, 8), (10, 9), (11, 10), (12, 1),
    #                 (13, 12), (14, 13), (15, 14), (16, 1), (17, 16), (18, 17),
    #                 (19, 18)]

    N, M, T, V, C = data.shape # N, M, T, V, C

    index_1=0 #can be changed to select the video
    fileNameList,classNumList=pickle_load(pickleDir+'/{}_label.pkl'.format(part))

    filename=fileNameList[index_1]
    ClassNum=classNumList[index_1]
    ClassName=actionDic[int(ClassNum)+1]#since the classNum starts from 0

    video_0=data[index_1]
    x_all=video_0[0,:,:,0]
    y_all=video_0[0,:,:,1]
    z_all=video_0[0,:,:,2]    
    

    fig = plt.figure()
    ax = plt.axes(projection="3d")

    # ax.set_xlim(np.amin(x_all), np.amax(x_all))
    # ax.set_ylim(np.amin(y_all), np.amax(y_all))
    # ax.set_zlim(np.amin(z_all), np.amax(z_all))

    for frame in range(T):
        for person in range(M):
            skeleton=video_0[person,frame]
            if skeleton.sum()==0:
                continue
            for bone in inward_ori_index:
                a,b=[i-1 for i in list(bone)]
                
                x_points = [skeleton[a,0],skeleton[b,0]]
                y_points = [skeleton[a,1],skeleton[b,1]]
                z_points = [skeleton[a,2],skeleton[b,2]]
                
                ax.plot3D(x_points, y_points, z_points,zdir='z', c='black')
                #ax.plot3D(x_points, z_points,y_points, 'black')    

            x_points = skeleton[:,0]
            y_points = skeleton[:,1]
            z_points = skeleton[:,2]
            ax.scatter3D(x_points, y_points, z_points,zdir='z', c='black')
            for i in range(V):
                label = str(i)
                #ax.text(x_points[i], y_points[i], z_points[i], label, zdir=None)
            #ax.scatter3D(x_points, z_points,y_points, c='black')

        ax.text2D(0.05, 0.95, '{}'.format(filename), transform=ax.transAxes)
        ax.text2D(0.95, 0.95, '{}'.format(ClassName), transform=ax.transAxes)
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_zlim(-1, 1)
        

        ax.set_xlabel('X axis')
        ax.set_ylabel('Y axis')
        ax.set_zlabel('Z axis')

        plt.draw()  
        plt.pause(.1)
        ax.cla()
    plt.close()

def pre_normalization(data,out_folder,actionDic,part='__', zaxis=[1, 19], xaxis=[7, 3]):#zaxis=[0, 1], xaxis=[8, 4])
    N, C, T, V, M = data.shape
    s = np.transpose(data, [0, 4, 2, 3, 1])  # N, C, T, V, M  to  N, M, T, V, C
    
    print('pad the null frames with the previous frames')
    for i_s, skeleton in enumerate(tqdm(s)):  # pad
        if skeleton.sum() == 0:
            print(i_s, ' has no skeleton')
        for i_p, person in enumerate(skeleton):
            if person.sum() == 0:
                continue
            if person[0].sum() == 0:
                index = (person.sum(-1).sum(-1) != 0)
                tmp = person[index].copy()
                person *= 0
                person[:len(tmp)] = tmp
            for i_f, frame in enumerate(person):
                if frame.sum() == 0:
                    if person[i_f:].sum() == 0:
                        rest = len(person) - i_f
                        num = int(np.ceil(rest / i_f))
                        pad = np.concatenate([person[0:i_f] for _ in range(num)], 0)[:rest]
                        s[i_s, i_p, i_f:] = pad
                        break
    #print('downsample to 10fps)')
    # visualization_skeleton_all(s,out_folder,actionDic)
    # frames_100=list(range(0,300,3))
    # T=len(frames_100)
    # assert T==100 #check if there is only 100 frames
    # s=s[:,:,frames_100,:,:] #resampling/down sampling

    T,s=first_downsample_method_100frames(s)  #for 10fps with 100 frames in first 300 frames of f_p
    # T,s=first_downsample_method_300frames(s) #with 10fps done for 300 frames
    # T,s=average_sample(s,3) #function only works atm for kernal_size=3
    


    print('sub the center joint #1 (spine joint in ntu and neck joint in kinetics)')
    for i_s, skeleton in enumerate(tqdm(s)):
        if skeleton.sum() == 0:
            continue
        main_body_center = skeleton[0][:, 1:2, :].copy()
        for i_p, person in enumerate(skeleton):
            if person.sum() == 0:
                continue
            mask = (person.sum(-1) != 0).reshape(T, V, 1)
            s[i_s, i_p] = (s[i_s, i_p] - main_body_center) * mask


    print('downsample and/or averaging/LPF')
    # visualization_skeleton_all(s,out_folder,actionDic)

    # T,s=LPF_sample(s,5) #function only works atm for kernal_size=5

    # T,s=average_sample(s,3) #unction only work for kernal_size=3 atm
    
    # visualization_skeleton_all(s,out_folder,actionDic)

    # T,s=LPF_sample_with_10FPS(s,5) #works only for 5 atm



    #visualization_skeleton_one(s,out_folder,actionDic,part)
    print('parallel the bone between hip(jpt 0) and spine(jpt 1) of the first person to the z axis')
    for i_s, skeleton in enumerate(tqdm(s)): #skeleton : M, T, V, C
        if skeleton.sum() == 0:
            continue
        joint_bottom = skeleton[0, 0, zaxis[0]] # coordinates of x,y,z (as seen on the 3d plot)
        joint_top = skeleton[0, 0, zaxis[1]]
        axis = np.cross(joint_top - joint_bottom, [0, 0, 1])
        angle = angle_between(joint_top - joint_bottom, [0, 0, 1])
        matrix_z = rotation_matrix(axis, angle)
        for i_p, person in enumerate(skeleton):
            if person.sum() == 0:
                continue
            for i_f, frame in enumerate(person):
                if frame.sum() == 0:
                    continue
                for i_j, joint in enumerate(frame): # goes through all 25 joints
                    s[i_s, i_p, i_f, i_j] = np.dot(matrix_z, joint)
    #visualization_skeleton_all(s,out_folder,actionDic,part)
    print(
        'parallel the bone between right shoulder(jpt 8) and left shoulder(jpt 4) of the first person to the x axis')
    for i_s, skeleton in enumerate(tqdm(s)):
        if skeleton.sum() == 0:
            continue
        joint_rshoulder = skeleton[0, 0, xaxis[0]]
        joint_lshoulder = skeleton[0, 0, xaxis[1]]
        axis = np.cross(joint_rshoulder - joint_lshoulder, [1, 0, 0])
        angle = angle_between(joint_rshoulder - joint_lshoulder, [1, 0, 0])
        matrix_x = rotation_matrix(axis, angle)
        for i_p, person in enumerate(skeleton):
            if person.sum() == 0:
                continue
            for i_f, frame in enumerate(person):
                if frame.sum() == 0:
                    continue
                for i_j, joint in enumerate(frame):
                    s[i_s, i_p, i_f, i_j] = np.dot(matrix_x, joint)
    # visualization_skeleton_all(s,out_folder,actionDic )
    data = np.transpose(s, [0, 4, 2, 3, 1])
    return data



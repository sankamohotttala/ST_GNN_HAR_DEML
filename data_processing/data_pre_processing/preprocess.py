import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.extend(['../'])
from rotation import *
from tqdm import tqdm

def visualization_skeleton(data):

    # inward_ori_index = [(1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5), (7, 6),
    #                 (8, 7), (9, 21), (10, 9), (11, 10), (12, 11), (13, 1),
    #                 (14, 13), (15, 14), (16, 15), (17, 1), (18, 17), (19, 18),
    #                 (20, 19), (22, 23), (23, 8), (24, 25), (25, 12)]


    inward_ori_index = [(1, 2), (2, 20), (3,20), (4, 20), (5, 4), (6, 5),
                    (7, 6), (8, 20), (9, 8), (10, 9), (11, 10), (12, 1),
                    (13, 12), (14, 13), (15, 14), (16, 1), (17, 16), (18, 17),
                    (19, 18)]

    N, M, T, V, C = data.shape # N, M, T, V, C
    
    video_0=data[4]
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
                ax.text(x_points[i], y_points[i], z_points[i], label, zdir=None)
            #ax.scatter3D(x_points, z_points,y_points, c='black')
        
        ax.set_xlim(-.5, .5)
        ax.set_ylim(-.5, .5)
        ax.set_zlim(-1, 1)
        

        ax.set_xlabel('X axis')
        ax.set_ylabel('Y axis')
        ax.set_zlabel('Z axis')

        plt.draw()
        plt.pause(.1)
        ax.cla()

def pre_normalization(data, zaxis=[1, 19], xaxis=[7, 3]):
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
    # visualization_skeleton(s)                    
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
    #visualization_skeleton(s) 
    print('parallel the bone between hip(jpt 0) and spine(jpt 1) of the first person to the z axis')
    for i_s, skeleton in enumerate(tqdm(s)):
        if skeleton.sum() == 0:
            continue
        joint_bottom = skeleton[0, 0, zaxis[0]]
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
                for i_j, joint in enumerate(frame):
                    s[i_s, i_p, i_f, i_j] = np.dot(matrix_z, joint)
    #visualization_skeleton(s) 
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
    # visualization_skeleton(s)
    data = np.transpose(s, [0, 4, 2, 3, 1])

    return data


if __name__ == '__main__':
    data = np.load('../data/ntu/xview/val_data.npy')
    pre_normalization(data)
    np.save('../data/ntu/xview/data_val_pre.npy', data)

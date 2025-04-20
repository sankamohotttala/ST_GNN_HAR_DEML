#test code to rotate the 3D structure in tests
from re import S
import sys
import numpy as np
from rotation_1 import *
import matplotlib.pyplot as plt
import math as maths #not useful
import tqdm

frames=50

def T_structure(frames):
    #numpy array of 3D points of 4 places
    #T, V, C shape

    a=np.array([[-1,3,6]]) #x,y,z as in visualization
    b=np.array([[5,3,6]])
    c=np.array([[11,3,6]])
    d=np.array([[5,3,0]])
    structure=np.concatenate((a, b,c,d), axis=0)
    video=np.zeros((frames,4,3))
    for i in range(frames):
        video[i,:,:]=structure
            
    return video.astype('float32')

def LongT_structure(frames):
    #numpy array of 3D points of 5 places
    #T, V, C shape

    a=np.array([[-1,3,12]]) #x,y,z as in visualization
    b=np.array([[5,3,12]])
    c=np.array([[11,3,12]])
    d=np.array([[5,3,6]])
    e=np.array([[5,3,0]])
    structure=np.concatenate((a, b,c,d,e), axis=0)
    structure=structure-d #centering the structure at 000
    video=np.zeros((frames,5,3))
    for i in range(frames):
        video[i,:,:]=structure
            
    return video.astype('float32')   



def visualization_structure(data):
    
    #inward_ori_index = [(1,2),(3,2),(4,2)]          # for T_structure
    inward_ori_index = [(1,2),(3,2),(4,2),(5,4)]    # for LongT_structure    
    
    T, V, C = data.shape #  T, V, C

    video_0=data

    fig = plt.figure()
    ax = plt.axes(projection="3d")

    # ax.set_xlim(np.amin(x_all), np.amax(x_all))
    # ax.set_ylim(np.amin(y_all), np.amax(y_all))
    # ax.set_zlim(np.amin(z_all), np.amax(z_all))

    for frame in range(T):
        
        skeleton=video_0[frame]
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
        ax.scatter3D(0, 0, 0,zdir='z', c='red')                             # for 0,0,0 coordiante
        for i in range(V):
            label = str(i)
            cordi_label='x:{0:.3f}, y:{1:.3f}, z:{2:.3f}'.format(x_points[i],y_points[i],z_points[i])
            ax.text(x_points[i], y_points[i], z_points[i], label, zdir=None)
            #ax.text(x_points[i], y_points[i], z_points[i],cordi_label , zdir=None)
        #ax.scatter3D(x_points, z_points,y_points, c='black')

        ax.text2D(0.05, 0.95, '{}'.format('t structure'), transform=ax.transAxes)
        # ax.text2D(0.95, 0.95, '{}'.format(ClassName), transform=ax.transAxes)
        ax.text2D(0.05, 0.05, 'frameNo: {}'.format(frame), transform=ax.transAxes)
        ax.set_xlim(-20, 20)
        ax.set_ylim(-20, 20)
        ax.set_zlim(-20, 20)
        

        ax.set_xlabel('X axis')
        ax.set_ylabel('Y axis')
        ax.set_zlabel('Z axis')
        



        if frame==0 or True: # this is used so that we can get the first frame and only first frame : to analyse the skeleton orientation 
            plt.draw()
            plt.pause(.1)   
            ax.cla() # put the breakpoint here
        else:
            pass
    plt.close()

def rotate(structure):
    edge=[0,2]
    if structure.sum() == 0: # T,V,C
        return -1
    joint_bottom = structure[0, edge[0]] # coordinates of x,y,z (as seen on the 3d plot)
    joint_top = structure[0, edge[1]]
    axis = np.cross(joint_top - joint_bottom, [0, 0, 1])
    angle = angle_between(joint_top - joint_bottom, [0, 0,1])
    matrix_z = rotation_matrix(axis, angle)

    for i_f, frame in enumerate(structure):
        if frame.sum() == 0:
            continue
        for i_j, joint in enumerate(frame): # goes through all 4 joints
            structure[i_f, i_j] = np.dot(matrix_z, joint)
    return structure

if __name__=='__main__':

    #video=T_structure(frames)
    video=LongT_structure(frames)
    visualization_structure(video)
    new_video=rotate(video)
    visualization_structure(new_video)  
    

    
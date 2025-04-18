#for Feature extraction  code
#later added: initial values are given is a new way it reduces the chances of mismatching the values and time is saved
#later added: label smoothing as an option with the loss function and smoothing value as an input 
#later added: learning rate scheduler with exponential decay but constant lr implementation is different compare to my previous codes
#later added: new stgcn code -> stgcn_1_edit.py <- with this we can set the initial weights and weight decay from this code with INITIALIZERS class
#later added: new stgcn code -> stgcn_1_edit_withoutinitclass.py <- this is the default code we used before

# from model.stgcn_original_similar import Model #use this for original ST-GCN - from 2018 paper
from model.stgcn_default_randomize_2 import Model, Model2,Model_randomzied, Model2_single_layer, Model2_new
from model.stgcn_default_randomize_2 import INITIALIZERS
import utility_custom as util #new library
import tensorflow as tf
from tqdm import tqdm
import argparse
import inspect
import shutil
import yaml
import os

import matplotlib
matplotlib.use("Agg")#remove this if you want to see the plots

#new libraries
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from pathlib import Path
import csv
import timeit
import pandas as pd

from sklearn.manifold import TSNE
from matplotlib.colors import ListedColormap
from sklearn.calibration import calibration_curve


def visualization_skeleton(data):
    #some lines were removed..no effect on the output
    #removed to make it more readable
    # inward_ori_index = [(1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5), (7, 6),
    #                 (8, 7), (9, 21), (10, 9), (11, 10), (12, 11), (13, 1),
    #                 (14, 13), (15, 14), (16, 15), (17, 1), (18, 17), (19, 18),
    #                 (20, 19), (22, 23), (23, 8), (24, 25), (25, 12)]

    inward_ori_index = [(1, 2), (2, 20), (3,20), (4, 20), (5, 4), (6, 5),
                    (7, 6), (8, 20), (9, 8), (10, 9), (11, 10), (12, 1),
                    (13, 12), (14, 13), (15, 14), (16, 1), (17, 16), (18, 17),
                    (19, 18)]

    N, C, T, V, M = data.shape # N, M, T, V, C
    data = tf.transpose(data, perm=[0, 4, 2, 3, 1]) 
    N, M, T, V, C = data.shape
    for i in range(N):  
        video_0=data[i]
        fig = plt.figure()
        ax = plt.axes(projection="3d")
        color=['black','blue']
        for frame in range(T):
            for person in range(M):
                skeleton=video_0[person,frame]
                skeleton=skeleton.numpy()
                if skeleton.sum()==0:
                    continue
                for bone in inward_ori_index:
                    a,b=[i-1 for i in list(bone)]
                    
                    x_points = [skeleton[a,0],skeleton[b,0]]
                    y_points = [skeleton[a,1],skeleton[b,1]]
                    z_points = [skeleton[a,2],skeleton[b,2]]
                    
                    ax.plot3D(x_points, y_points, z_points,zdir='z', c=color[person])
                    
                x_points = skeleton[:,0]
                y_points = skeleton[:,1]
                z_points = skeleton[:,2]
                ax.scatter3D(x_points, y_points, z_points,zdir='z', c=color[person])
            plt.draw()
            plt.pause(.1)
            ax.cla()
    

def get_parser():
    # parameter priority: command line > config > default
    parser = argparse.ArgumentParser(
        description='Spatial Temporal Graph Convolutional Neural Network for Skeleton-Based Action Recognition')
    parser.add_argument(
        '--base-lr', type=float, default=base_lr, help='initial learning rate')
    parser.add_argument(
        '--num-classes', type=int, default=ChildDatasetClasses, help='number of classes in dataset')
    parser.add_argument(
        '--batch-size', type=int, default=batchSize, help='training batch size')
    parser.add_argument(
        '--num-epochs', type=int, default=used_epochs, help='total epochs to train')
    parser.add_argument(
        '--save-freq', type=int, default=2, help='periodicity of saving model weights')
    parser.add_argument(
        '--checkpoint-path',
        '--checkpoint-path',                                                                                                                          
        default=checkpoint_restore_path,
        help='folder to store model weights')
    parser.add_argument(
        '--log-dir',
        default=logDirectory,
        help='folder to store model-definition/training-logs/hyperparameters')
    parser.add_argument(
        '--train-data-path',
        default=train_data_path,
        help='path to folder with training dataset tfrecord files')
    parser.add_argument(
        '--test-data-path',
        default=test_data_path,
        help='path to folder with testing dataset tfrecord files')
    parser.add_argument(
        '--steps',
        type=int,
        default=steps,
        nargs='+',
        help='the epoch where optimizer reduce the learning rate, eg: 10 50')
    parser.add_argument(
        '--gpus',
        default=None,                                                    
        nargs='+',
        help='list of gpus to use for training, eg: "/gpu:0" "/gpu:1"')
#["GPU:0", "GPU:1"],
#["/gpu:0", "/gpu:1"]
    parser.add_argument(
        '--iteration',
        default=iterationNum,                                                    
        type=int,
        help='iterations per epoch when batch size is 1[not sure about this explanation]')  #originally 40000
        
    return parser

def visualization_skeleton(data):
    #some lines were removed..no effect on the output
    #removed to make it more readable
    # inward_ori_index = [(1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5), (7, 6),
    #                 (8, 7), (9, 21), (10, 9), (11, 10), (12, 11), (13, 1),
    #                 (14, 13), (15, 14), (16, 15), (17, 1), (18, 17), (19, 18),
    #                 (20, 19), (22, 23), (23, 8), (24, 25), (25, 12)]

    inward_ori_index = [(1, 2), (2, 20), (3,20), (4, 20), (5, 4), (6, 5),
                    (7, 6), (8, 20), (9, 8), (10, 9), (11, 10), (12, 1),
                    (13, 12), (14, 13), (15, 14), (16, 1), (17, 16), (18, 17),
                    (19, 18)]

    N, C, T, V, M = data.shape # N, M, T, V, C
    data = tf.transpose(data, perm=[0, 4, 2, 3, 1]) 
    N, M, T, V, C = data.shape
    for i in range(N):  
        video_0=data[i]
        fig = plt.figure()
        ax = plt.axes(projection="3d")
        color=['black','blue']
        for frame in range(T):
            for person in range(M):
                skeleton=video_0[person,frame]
                skeleton=skeleton.numpy()
                if skeleton.sum()==0:
                    continue
                for bone in inward_ori_index:
                    a,b=[i-1 for i in list(bone)]
                    
                    x_points = [skeleton[a,0],skeleton[b,0]]
                    y_points = [skeleton[a,1],skeleton[b,1]]
                    z_points = [skeleton[a,2],skeleton[b,2]]
                    
                    ax.plot3D(x_points, y_points, z_points,zdir='z', c=color[person])
                    
                x_points = skeleton[:,0]
                y_points = skeleton[:,1]
                z_points = skeleton[:,2]
                ax.scatter3D(x_points, y_points, z_points,zdir='z', c=color[person])
            ax.set_xlim(-1, 1)
            ax.set_ylim(-1, 1)
            ax.set_zlim(-1, 1)
            plt.draw()
            plt.pause(.1)
            ax.cla()
    
def save_arg(arg):
    # save arg
    arg_dict = vars(arg)
    if not os.path.exists(arg.log_dir):
        os.makedirs(arg.log_dir)
    with open(os.path.join(arg.log_dir, "config.yaml"), 'w') as f:
        yaml.dump(arg_dict, f)



def get_dataset(directory, num_classes=60, batch_size=32, drop_remainder=False,
                shuffle=False, shuffle_size=1000): #previosuly num_classes was 60;i chanegd to fiund the reason for error
    # dictionary describing the features.
    feature_description = {
        'features': tf.io.FixedLenFeature([], tf.string),
        'label'     : tf.io.FixedLenFeature([], tf.int64)
    }

    # parse each proto and, the features within
    def _parse_feature_function(example_proto):
        features = tf.io.parse_single_example(example_proto, feature_description)
        data =  tf.io.parse_tensor(features['features'], tf.float32)
        label = tf.one_hot(features['label'], num_classes)
        data = tf.reshape(data, (3, 300, 20, 1)) #changed both num_joints from 25 to 20 and num_skeletons from 2 to 1
        return data, label

    records = [os.path.join(directory, file) for file in os.listdir(directory) if file.endswith("tfrecord")]
    dataset = tf.data.TFRecordDataset(records, num_parallel_reads=len(records))
    dataset = dataset.map(_parse_feature_function)
    dataset = dataset.batch(batch_size, drop_remainder=drop_remainder)
    dataset = dataset.prefetch(batch_size)
    if shuffle:
        dataset = dataset.shuffle(shuffle_size)
    return dataset




    # paramss.action=softmax_logits

#   strategy.run(step_fn, args=(features, labels,))

#------------------------------confusion matrix part----------------------
#-------------------------------------------------------------------------



def confusion_matrix_data(predictions,labels,test_iter):
    batch_size,class_num=labels.shape
    if test_iter==0:
        #iteration=np.zeros((class_num,1))
        #iteration_final=np.zeros((class_num,1)) non need
        #predict_val=[]
        #label_val=[]
        pass
    for i in range(batch_size):
        ind_pred=np.argmax(predictions[i])
        ind_label=np.argmax(labels[i])
        iteration[ind_label]+=1
        predict_val.append(ind_pred)
        label_val.append(ind_label)
    return predict_val,label_val,iteration

def confusion_matrix_draw(predict,label,iteration,test_ite):
    N=iteration.shape[0]
    iteration_final=np.zeros((N,1))
    classList=[str(i) for i in list(range(1,N+1))]
    
    for i in range(N):
        iteration_final[i]=(1/iteration[i])
    confusion_mtx = tf.math.confusion_matrix(label, predict,dtype=tf.dtypes.float32) 
    confusion_mtx=confusion_mtx*iteration_final
    confusion_mtx=confusion_mtx*100
    plt.figure(figsize=(10, 8))
    sns.heatmap(confusion_mtx,xticklabels=classList,yticklabels=classList,  annot=True, fmt='.2f')
    plt.xlabel('Prediction')
    plt.ylabel('Label')
    pathDirectoryCF=Path(folder_name+'/aditive_images/')
    if not (pathDirectoryCF.exists()):
        os.mkdir(pathDirectoryCF)
    #plt.savefig('images/conf{}.jpg'.format(test_ite))
    plt.savefig(os.path.join(pathDirectoryCF,'conf{}.jpg'.format(test_ite)))
    plt.close()


def confusion_matrix_data_2(predictions,labels,test_iter):
    batch_size,class_num=labels.shape
    if test_iter==0:
        #iteration=np.zeros((class_num,1))
        #iteration_final=np.zeros((class_num,1)) non need
        #predict_val=[]
        #label_val=[]
        pass
    for i in range(batch_size):
        ind_pred=np.argmax(predictions[i])
        ind_label=np.argmax(labels[i])
        iteration_2[ind_label]+=1
        predict_val_2.append(ind_pred)
        label_val_2.append(ind_label)
    return predict_val_2,label_val_2,iteration_2


def confusion_matrix_draw_2(predict,label,iteration,test_ite):
    N=iteration.shape[0]
    iteration_final=np.zeros((N,1))
    classList=[str(i) for i in list(range(1,N+1))]
    
    for i in range(N):
        iteration_final[i]=(1/iteration[i])
    confusion_mtx = tf.math.confusion_matrix(label, predict,dtype=tf.dtypes.float32) 
    confusion_mtx=confusion_mtx*iteration_final
    confusion_mtx=confusion_mtx*100
    plt.figure(figsize=conf_fig_size) #(60, 48) changed to 10,8
    sns.heatmap(confusion_mtx,xticklabels=classList,yticklabels=classList,  annot=conf_annotation, fmt='.2f')
    plt.xlabel('Prediction')
    plt.ylabel('Label')
    pathDirectoryCF=Path(folder_name+'/images_independant/')
    if not (pathDirectoryCF.exists()):
        os.mkdir(pathDirectoryCF)
    #plt.savefig('images/conf{}.jpg'.format(test_ite))
    plt.savefig(os.path.join(pathDirectoryCF,'conf{}.jpg'.format(test_ite)))
    plt.close()

#------------------------------End Confusion Matrix Part----------------------
#-------------------------------------------------------------------------
def save_values_csv(predictions,labels,features_data,epoch):
    #create the csv file

    batch_size,class_num=labels.shape
    if test_iter==0:
        pass
    for i in range(batch_size):
        dataTmp=[]
        uniqueID=features_data[i].numpy().sum()
        ind_pred=np.argmax(predictions[i]) #predicted class
        ind_label=np.argmax(labels[i]) #from 0 to n-1 ; real class
        probabilityArray=list(predictions[i].numpy())
        max_probability=np.amax(predictions[i]) # predicted class probability
        assert predictions[i][ind_pred]== max_probability
        real_class_proba=predictions[i].numpy()[ind_label]#real class prbability
        if save_all_data:
            dataTmp.extend([uniqueID,ind_label,real_class_proba,ind_pred,max_probability]+probabilityArray)
        else:
            dataTmp.extend([uniqueID,ind_label,real_class_proba,ind_pred,max_probability])
        csvData.append(dataTmp)

def tsne_create(embeddings,labels):
    tmp_labels=[]
    tmp_embed=[]
    for sample in embeddings:
        for i in range(sample.shape[0]):
            tmp_embed.append(sample[i])
    for label in labels:
        for i in range(label.shape[0]):
            index=np.argmax(label[i])
            tmp_labels.append(index)

    tmp_embed_np=np.asarray(tmp_embed)
    tmp_labels_np=np.asarray(tmp_labels)
    tsne_embedded = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=3).fit_transform(tmp_embed_np)
    # print(tsne_embedded.shape)
    return tsne_embedded,tmp_labels_np


def plotAccuracy(train_epoch,test_epoch,cross_loss,cross_loss_2,epochsNumber,train_ite,lr,tsne_out,labels_out,final=True):
    epochsNum = range(epochsNumber)
    train_iteration = range(epochsNumber)
    test_iteration = range(epochsNumber)
    train_real_iteration=range(train_ite)

    classStr=['cls - '+str(i) for i in list(range(num_classes))]
    pathHome=Path(folder_name+'/matplotlib_graphs/')
    if not (pathHome.exists()):
        os.mkdir(pathHome)

    if not final:
        pathDirectoryCF=Path(str(pathHome)+f'/{epochsNumber}/')
        if not (pathDirectoryCF.exists()):
            os.mkdir(pathDirectoryCF)
    else:
        pathDirectoryCF=pathHome

    plt.figure()
    plt.plot(epochsNum, train_epoch, 'k', label='Training epoch accuracy')
    plt.title('Training accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuaracy')
    #plt.legend()
    plt.savefig(os.path.join(pathDirectoryCF,'Train_plot.jpg'))
    plt.close()

    plt.figure()
    plt.plot(epochsNum, test_epoch, 'k', label='Test epoch accuracy')
    plt.title('Testing accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuaracy')
    #plt.legend()
    plt.savefig(os.path.join(pathDirectoryCF,'Test_plot.jpg'))
    plt.close()

    plt.figure()
    plt.plot(epochsNum, train_epoch, 'b', label='Training epoch accuracy') 
    plt.plot(epochsNum, test_epoch, 'g', label='Test epoch accuracy')
    plt.title('Training and Testing accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuaracy')
    plt.legend()
    #plt.show()
    plt.savefig(os.path.join(pathDirectoryCF,'Train_test_plot.jpg'))
    plt.close()

    plt.figure()
    plt.plot(train_iteration, cross_loss, 'k', label='cross entropy loss')
    plt.title('Cross Entropy Loss - Training')
    plt.xlabel('epochs')
    plt.ylabel('Loss')
    #plt.legend()
    plt.savefig(os.path.join(pathDirectoryCF,'cross_loss_plot.jpg'))
    plt.close()
    
    plt.figure()
    plt.plot(test_iteration, cross_loss_2, 'k', label='cross entropy loss')
    plt.title('Cross Entropy Loss - Test')
    plt.xlabel('epochs')
    plt.ylabel('Loss')
    #plt.legend()
    plt.savefig(os.path.join(pathDirectoryCF,'cross_loss_plot_val.jpg'))
    plt.close()

    plt.figure()
    plt.plot(train_iteration, cross_loss, 'b', label='Training loss') 
    plt.plot(test_iteration, cross_loss_2, 'g', label='Test loss ')
    plt.title('Training and Testing Cross entropy loss')
    plt.xlabel('epochs')
    plt.ylabel('Loss')
    plt.legend()
    #plt.show()
    plt.savefig(os.path.join(pathDirectoryCF,'Train_test_cross_entropy_plot.jpg'))
    plt.close()

    #for learning rate plot
    fig=plt.figure()
    ax1 = fig.add_subplot(111)
    ax2 = ax1.twiny()

    ax1.plot(train_real_iteration, lr, 'k', label='learning_rate')
    ax1.set_title('Learning rate')
    ax1.set_xlabel('iterations')
    ax1.set_ylabel('value')
    ax1.grid(True)

    ax2.set_xticks(ax1.get_xticks())
    ax2.set_xbound(ax1.get_xbound())
    ax2.set_xticklabels([round(x*(epochsNumber/train_ite), 2) for x in ax1.get_xticks()])
    ax2.set_xlabel('epochs')
    #plt.legend()
    fig.savefig(os.path.join(pathDirectoryCF,'learning rate.jpg'))
    plt.close()

    #for tsne plot
    plt.figure(figsize=tsne_fig_size)
    colors = plt.cm.rainbow(np.linspace(0, 1, num_classes))
    for label_id in range(num_classes):
        i = np.where(labels_out == label_id)
        plt.scatter(tsne_out[i, 0], tsne_out[i, 1], color=colors[label_id,:])
    plt.legend(classStr, loc='best')
    # plt.show()
    plt.xlabel('componant 1')
    plt.ylabel('componant 2')
    plt.savefig(os.path.join(pathDirectoryCF,'t-sne visualiz.jpg'))
    plt.close()


def reliabilityDigram(dataframe, class_num, epoch_num, bin_n=10):

    BINS = bin_n               #number of bins
    # CLASS = class_num          #from 0 to N-1

    pathDirectoryRD=os.path.join(folder_name,'reliability_diagrams')
    if not os.path.exists(pathDirectoryRD):
        os.makedirs(pathDirectoryRD)

    pathDirectoryRD_epoch=os.path.join(pathDirectoryRD,'epoch_'+str(epoch_num+1))
    if not os.path.exists(pathDirectoryRD_epoch):
        os.makedirs(pathDirectoryRD_epoch)

    for i in range(class_num):

        CLASS=i

        # df =pd.read_csv(pathCSV)
        df = dataframe

        # this is for binwise dictionary data structure
        bin_dic={i:[] for i in range(1,BINS+1)} #start from 1 fo to BINS

        y_pred=[]
        y_true=[]
        for i in range(df.shape[0]):
            values=df.iloc[i,5:].values # as a numpy array (N,)
            ground_truth=df.iloc[i,1]
            predict_class=df.iloc[i,3]
            classify=df.iloc[i,1]==df.iloc[i,3]
            tmp_dic={}
            tmp_dic['classify']=classify
            tmp_dic['ground_truth']=ground_truth
            tmp_dic['predict_class']=predict_class
            tmp_dic['values']=values


            expected_class_prob=values[CLASS]
            y_pred.append(expected_class_prob)
            if ground_truth==CLASS:
                y_true.append(1)
            else:
                y_true.append(0)
            # y_true.append(ground_truth)

        y_pred_np=np.array(y_pred)
        y_true_np=np.array(y_true)
        prob_true, prob_pred = calibration_curve(y_true_np, y_pred_np, n_bins=BINS)

        plt.grid()

        xx=prob_pred
        yy=prob_true
        plt.plot(xx , yy,'o--')

        line_val=np.array([i for i in range(11)])/10
        
        plt.plot(line_val , line_val,'--',color='black')    
        plt.title('Reliability Diagram')
        plt.xlabel('Average Probability of sample bin')
        plt.ylabel('Frequency of Ground Truth')
        plt.savefig(os.path.join(pathDirectoryRD_epoch,f'reliability_class_id_{CLASS}.png'))
        # plt.show()
        plt.grid('off')
        plt.close()

def reliabilityDigram_2(dataframe, class_num, epoch_num, bin_n=10):


    BINS = bin_n               #number of bins
    average_plot=True

    pathDirectoryRD=os.path.join(folder_name,'reliability_diagrams_2')
    if not os.path.exists(pathDirectoryRD):
        os.makedirs(pathDirectoryRD)

    pathDirectoryRD_epoch=os.path.join(pathDirectoryRD,'epoch_'+str(epoch_num+1))
    if not os.path.exists(pathDirectoryRD_epoch):
        os.makedirs(pathDirectoryRD_epoch)

    pred_list=[]
    true_list=[]
    for i in range(class_num):

        CLASS=i
        
        bin_edge=[0]
        bin_edge.extend([i/BINS for i in range(1,BINS+1)])
        bin_edge[BINS] += 1e-10 #for last bin for edge case

        df =dataframe

        # this is for binwise dictionary data structure
        bin_dic={i:[] for i in range(1,BINS+1)} #start from 1 fo to BINS
        for i in range(df.shape[0]):
            values=df.iloc[i,5:].values # as a numpy array (N,)
            ground_truth=df.iloc[i,1]
            predict_class=df.iloc[i,3]
            classify=df.iloc[i,1]==df.iloc[i,3]
            tmp_dic={}
            tmp_dic['classify']=classify
            tmp_dic['ground_truth']=ground_truth
            tmp_dic['predict_class']=predict_class
            tmp_dic['values']=values

            expected_class_prob=values[CLASS]
            for j in range(1,BINS+1):                    #assign the correct bin
                if expected_class_prob<bin_edge[j] and expected_class_prob>=bin_edge[j-1]:
                    #add to value correctly to the jth bin
                    bin_dic[j].append(tmp_dic)
                    break
            
        # frquency of Ground Truth == CLASS in each bin
        pred_bins={i:len(bin_dic[i]) for i in range(1,BINS+1)}
        ground_truth_bins={i:0 for i in range(1,BINS+1)}
        for i in range(1,BINS+1):
            for j in range(len(bin_dic[i])):
                _=bin_dic[i][j]['ground_truth'] #for debugging
                if bin_dic[i][j]['ground_truth']==CLASS:
                    ground_truth_bins[i]+=1

        frequency_bins={i:0 for i in range(1,BINS+1)}
        for i in range(1,BINS+1):
            if pred_bins[i]!=0:
                frequency_bins[i]=ground_truth_bins[i]/pred_bins[i]
            #no need for else since it is already 0 ; 

        # avergae probability of class in each bin
        average_prob_bins={i:0 for i in range(1,BINS+1)}
        for i in range(1,BINS+1):
            for j in range(len(bin_dic[i])):
                average_prob_bins[i]+=bin_dic[i][j]['values'][CLASS]
            if len(bin_dic[i])!=0:
                average_prob_bins[i]/=len(bin_dic[i])
            else:
                average_prob_bins[i] = ( bin_edge[i]+bin_edge[i-1] ) / 2
                #mid probability if no samples in the bin


        # xx=[i for i in average_prob_bins.values()]
        # yy=[j for j in frequency_bins.values()]

        xx=[i for i in average_prob_bins.values()]
        yy=[j for j in frequency_bins.values()]

        pred_list.append(xx)
        true_list.append(yy)

        plt.plot(xx , yy,'o--')

        line_val=np.array([i for i in range(11)])/10
        
        plt.plot(line_val , line_val,'--',color='black')    

        plt.grid()
        plt.title('Reliability Diagram')
        plt.xlabel('Average Probability of sample bin')
        plt.ylabel('Frequency of Ground Truth')
        # plt.savefig(os.path.join('reliability_diagrams',f'reliability_diagram_{CLASS}.png'))
        plt.savefig(os.path.join(pathDirectoryRD_epoch,f'reliability_clss_id_{CLASS}.png'))
        plt.grid('off')
        # plt.show()
        plt.close()

    if average_plot == True:

        pred_average=[]
        true_average=[]
        
        for bin in range(BINS):
            tmp=0
            tmp_1=0
            for id in range(class_num):
                tmp += pred_list[id][bin]
                tmp_1 += true_list[id][bin]
            pred_average.append(tmp/class_num)
            true_average.append(tmp_1/class_num)
            
        xx_=pred_average
        yy_=true_average

        line_val_n=np.array([i for i in range(11)])/10
        plt.plot(line_val_n , line_val_n,'--',color='blue')

        plt.plot(xx_ , yy_,'ko--',label='average')
        plt.grid()
        plt.title('Reliability Diagram')
        plt.xlabel('Average Probability of sample bin')
        plt.ylabel('Frequency of Ground Truth')
        plt.legend()
        plt.savefig(os.path.join(pathDirectoryRD,f'RD_averaged_epoch_{epoch_num+1}.png'))
        # plt.show()
        plt.close()        

def plotBoxWhisker(dataframe, epoch_num):

    pathDirectoryBW=os.path.join(folder_name,'box_wisker')
    if not os.path.exists(pathDirectoryBW):
        os.makedirs(pathDirectoryBW)

    pathDirectoryBW_epoch=os.path.join(pathDirectoryBW,'epoch_'+str(epoch_num+1))
    if not os.path.exists(pathDirectoryBW_epoch):
        os.makedirs(pathDirectoryBW_epoch)

    df=dataframe
    #chnage  colunm names at two places

    for status in range(4):
        
        type_name={0:'all_data',1:'correctly',2:'incorrectly',3:'both_distrib'}
        trial=type_name[status]

        if status==0: #all data
            final_df=df

        elif status==1: #only correctly classified
            final_df=df.loc[df['real_label']==df['predicted_label']]
        elif status==2: #only incorrectly classified
            final_df=df.loc[df['real_label']!=df['predicted_label']]
        elif status==3: # both misclassify and classify in same plot
            newList=newListClassify(df)
            df['Result']=newList
            final_df=df

        sns.set(style='whitegrid')

        if status!=3:
            sns.boxplot(x='real_label',y='real_probability',data=final_df,palette='Set3').set(xlabel='label',ylabel='probability')#,dodge=True
            plt.savefig(os.path.join(pathDirectoryBW_epoch,f'BW_{status}_{trial}_epoch{epoch_num+1}.jpg'))
            plt.close()
            # seaborn.despine(offset=10, trim=True)
        else:
            sns.boxplot(x='real_label',y='real_probability',hue='Result',data=final_df,palette='Set3',dodge=True).set(xlabel='label',ylabel='probability')#,dodge=True
            plt.savefig(os.path.join(pathDirectoryBW_epoch,f'BW_{status}_{trial}_epoch{epoch_num+1}.jpg'))
            plt.close()
        
            # seaborn.despine(offset=10, trim=True)
        
        # plt.show()        
        #plt.savefig('images/conf{}.jpg'.format(test_ite))


def newListClassify(dataframe):
    listCol=[]
    for index, row in dataframe.iterrows():
        if row['real_label'] == row['predicted_label']:
            listCol.append('Correct')
        else:
            listCol.append('Incorrect')
    
    return listCol


def save_var_new(ind,static_,config,config_2,*argv):#ImplementationName,train_data_path,test_data_path,isStepDecay,trials
    tmp=f'trial id: {ind}\n'
    if static_==True: #static values used in all trials
        for key,val in config_2[ind_2].items():
            tmp+=f'{key}: {val}\n'
    else: #dynamic values used in all trials
        for key,val in config_2[ind].items():
            tmp+=f'{key}: {val}\n'
    for key,val in config[ind].items():
        tmp+=f'{key}: {val}\n'
    for var in argv:
        tmp+=f'{var}\n'
    util.WriteFile(tmp,logDirectory,'variables')
    


#config dics
config = {1: {'schedular':'piecewise','decay':0.92,'loss_func':'category_cross','label_smooth_val':0,'scale':2., 'mode':'fan_out', 'dist':'truncated_normal','reg':'l2','reg_valu':0.001,'epochs_fc':3},
          2: {'schedular':'piecewise','decay':0.92,'loss_func':'category_cross','label_smooth_val':0,'scale':2., 'mode':'fan_out', 'dist':'truncated_normal','reg':'l2','reg_valu':0.001,'epochs_fc':3},
          3: {'schedular':'piecewise','decay':0.92,'loss_func':'category_cross','label_smooth_val':0,'scale':2., 'mode':'fan_out', 'dist':'truncated_normal','reg':'l2','reg_valu':0.001,'epochs_fc':3},
          4: {'schedular':'piecewise','decay':0.92,'loss_func':'category_cross','label_smooth_val':0,'scale':2., 'mode':'fan_out', 'dist':'truncated_normal','reg':'l2','reg_valu':0.001,'epochs_fc':3},
          5: {'schedular':'piecewise','decay':0.92,'loss_func':'category_cross','label_smooth_val':0,'scale':2., 'mode':'fan_out', 'dist':'truncated_normal','reg':'l2','reg_valu':0.001,'epochs_fc':3},
          6: {'schedular':'piecewise','decay':0.92,'loss_func':'category_cross','label_smooth_val':0,'scale':2., 'mode':'fan_out', 'dist':'truncated_normal','reg':'l2','reg_valu':0.001,'epochs_fc':3}
          }


#non looping config
# config_2 = {1: { 'target_classes': 15, 'source_cls':60,'proto':'S1','lr': 0.01,'steps':[10,20], 'optimizer':'SGD','batch_size': 4, 'epochs': 30, 'stgcn_out':9, 'FPS':30},
#             2: { 'target_classes': 10, 'source_cls':60,'proto':'S2','lr': 0.01,'steps':[10,20], 'optimizer':'SGD','batch_size': 4, 'epochs': 30, 'stgcn_out':9, 'FPS':30},
#             3: { 'target_classes': 10, 'source_cls':60,'proto':'S3','lr': 0.01,'steps':[10,20], 'optimizer':'SGD','batch_size': 4, 'epochs': 30, 'stgcn_out':9, 'FPS':30},
#             4: { 'target_classes': 5, 'source_cls':60,'proto':'S4','lr': 0.01,'steps':[10,20], 'optimizer':'SGD','batch_size': 4, 'epochs': 30, 'stgcn_out':9, 'FPS':30}
#             }
config_2 = {1: { 'target_classes': 5, 'source_cls':120,'proto':'S4','lr': 0.01,'steps':[15,25], 'optimizer':'SGD','batch_size': 4, 'epochs': 30, 'stgcn_out': 9,  'FPS':30},
            2: { 'target_classes': 5, 'source_cls':60,'proto':'S4','lr': 0.01,'steps':[15,25], 'optimizer':'SGD','batch_size': 4, 'epochs': 30, 'stgcn_out': 9,  'FPS':30},
            3: { 'target_classes': 5, 'source_cls':22,'proto':'S4','lr': 0.01,'steps':[15,25], 'optimizer':'SGD','batch_size': 4, 'epochs': 30, 'stgcn_out': 9,  'FPS':30},
            4: { 'target_classes': 5, 'source_cls':44,'proto':'S4','lr': 0.01,'steps':[15,25], 'optimizer':'SGD','batch_size': 4, 'epochs': 30, 'stgcn_out': 9, 'FPS':30},
            5: { 'target_classes': 5, 'source_cls':60,'proto':'S4','lr': 0.01,'steps':[15,25], 'optimizer':'SGD','batch_size': 4, 'epochs': 30, 'stgcn_out': 9,  'FPS':30},
            6: { 'target_classes': 5, 'source_cls':120,'proto':'S4','lr': 0.01,'steps':[15,25], 'optimizer':'SGD','batch_size': 4, 'epochs': 30, 'stgcn_out': 9, 'FPS':30}            
            }

config_2_static = False      #True if we are using the first instance in config_2 for all trials; False if we have similar instances in config_2 as in config
use_ind=3 # if above True, then use this index from config_2


#we should do experiment across CWBG protocols;not across  target datasets
root_folder='results_loop_random/'+'root'+ util.data_time_string()
save_all_data=True
skeleton_structure='coco' #need to probably change the graph as well

unFreezeFirst=False #since false , we freeze the first set of stgcn layers
# STGCN_OUT=8 #output from stgcn layer (starting from 1 NOT 0)

# weight_path=r'F:\Codes\joint attention\2022 - Journal\New TF and NTU-44  implementations\Create weights\save_weights\NTU 60\ntu60' #last ntu120 is necessary
weight_path_config = {1:r'F:\Codes\joint attention\2022 - Journal\New TF and NTU-44  implementations\Create weights\save_weights\NTU 120\CKPT 6\ntu120',
                      2:r'F:\Codes\joint attention\2022 - Journal\New TF and NTU-44  implementations\Create weights\save_weights\NTU 60\ntu60',
                      3:r'F:\Codes\joint attention\2022 - Journal\New TF and NTU-44  implementations\Create weights\save_weights\NTU 22\ntu22',
                      4:r'F:\Codes\joint attention\2022 - Journal\New TF and NTU-44  implementations\Create weights\save_weights\NTU 44 10FPS\ntu44_10fps',
                      5:r'F:\Codes\joint attention\2022 - Journal\New TF and NTU-44  implementations\Create weights\save_weights\NTU 60 10FPS\ntu60_10fps',
                      6:r'F:\Codes\joint attention\2022 - Journal\New TF and NTU-44  implementations\Create weights\save_weights\NTU 120 10FPS\ntu120_10fps'
                      }
trials=6
for ind in range(1,trials+1):
    ind_2 = ind if not config_2_static else use_ind #if True -> 1 else -> ind
    weight_path = weight_path_config[ind]

    initial_model=INITIALIZERS(scale=config[ind]['scale'], mode=config[ind]['mode'], distribution=config[ind]['dist'],reg=config[ind]['reg'],reg_value=config[ind]['reg_valu']) #l2 or l1 for reg;
    # original valuesINITIALIZERS(scale=2., mode="fan_out", distribution="truncated_normal",reg='l1',reg_value=0.001) 
    # unfreeze_start= config_2[ind_2]['S']  #unfreeze the layers starting from this one
    # unfreeze_end = config_2[ind_2]['E']  #unfreeze the layers ending at one before this one; ex: if 12 then at 11
    epochs_fc = config[ind]['epochs_fc'] #number of epochs for training the fully connected layers only
    STGCN_OUT=config_2[ind_2]['stgcn_out'] #output from stgcn layer (starting from 1 NOT 0)



    conf_fig_size=(10,8) #figure size for confusion matrix - small : (10,8), large : (64,48)
    conf_annotation=True #if true then the numbers will be shown in the confusion matrix
    tsne_fig_size=(8,8) # for small: (10,10) , for large: (25,25) , extra large: (30,30)

    loss_fuction=config[ind]['loss_func'] #softmax_cross or category_cross ->used for label smoothing
    label_smooth_val=config[ind]['label_smooth_val'] #0.1 is the default value; 0 is when no label smoothing done

    #LR and LR schedular selection 
    scedular_method=config[ind]['schedular'] #check below dictionary for the options
    scedular_dic={'piecewise':'piecewise_decay','expo':'expo_decay','constant':'constant'} 
    if scedular_dic[scedular_method]=='constant':
        ConstantBaseLearnngRate=True #if true then base_lr will be same ;
        scheduler='constant'
    else:
        ConstantBaseLearnngRate=False # if False LRschedular will be used as in the code
        scheduler=scedular_dic[scedular_method] #piecewise_decay or expo_decay
    decay_rate=config[ind]['decay'] #.96 default ,.5 ->e-9 value, .75 -> 0.00017..
    isStepDecay=True #if true step decay is used instead of constinous decay

    base_lr=config_2[ind_2]['lr'] #0.01
    steps=config_2[ind_2]['steps']
    opti_mode=config_2[ind_2]['optimizer'] #adam or sgd
    # steps=[1,5]

    PreTrainedClasses=config_2[ind_2]['source_cls'] #since no transfer learning here    
    ChildDatasetClasses=config_2[ind_2]['target_classes'] #change the train, val data paths as well

    ProtocolTraining=config_2[ind_2]['proto'] #S1, S2, S3, S4 ->full, similar, dissimilar, shared
    batchSize=config_2[ind_2]['batch_size']

    datatime=util.data_time_string() #used as a unique number as well so we know when the code was ran
    trial={'S1':'Full','S2':'Simi','S3':'Diss','S4':'Shar'} 
    LearningMode=trial[ProtocolTraining] 

    FPS=config_2[ind_2]['FPS'] #either 10 or 30
    #checkpoint_restore_path=None        #this is given a value below

    iteration_dic={'S1':924,'S2':612,'S3':628,'S4':312} #full, similar, dissimilar, shared
    iterationNum=iteration_dic[ProtocolTraining] #cwbg all - 924 # cwbg similar 612 , cwbg dissimilar 628


    if ConstantBaseLearnngRate==True:
        iterationNumPrint= -1   #since lr is constant
    else:
        iterationNumPrint=iterationNum

    used_epochs = config_2[ind_2]['epochs'] #30
    checkpoint_name='NEW_60' #folder name; not using these checkpoints anymore

    main_checkpoint=r'F:\Codes\joint attention\2022 - Journal\New TF and NTU-44  implementations\TFL_FX_FT_final\checkpoints' #not using these checkpoints anymore
    checkpoint_restore_path=os.path.join(main_checkpoint,checkpoint_name) #not using these checkpoints anymore

    # for CFBG dataset
    if ProtocolTraining =='S1':    #full 
        train_data_path = r'F:\Codes\joint attention\2022\visualize_vattu_child\save_directory\xsub\train_data'
        test_data_path  = r'F:\Codes\joint attention\2022\visualize_vattu_child\save_directory\xsub\val_data'
    elif ProtocolTraining =='S2': #similar
        train_data_path = r'F:\Codes\joint attention\2022\visualize_vattu_child\save_directory_2\xsub\train_data'
        test_data_path  = r'F:\Codes\joint attention\2022\visualize_vattu_child\save_directory_2\xsub\val_data'
    elif ProtocolTraining =='S3': #dissimilar
        train_data_path = r'F:\Codes\joint attention\2022\visualize_vattu_child\save_directory_3\xsub\train_data'
        test_data_path  = r'F:\Codes\joint attention\2022\visualize_vattu_child\save_directory_3\xsub\val_data'
    elif ProtocolTraining =='S4': #shared
        train_data_path = r'F:\Codes\joint attention\2022\visualize_vattu_child\save_directory_4\xsub\train_data'
        test_data_path  = r'F:\Codes\joint attention\2022\visualize_vattu_child\save_directory_4\xsub\val_data'
    else:
        ValueError()

    # ImplementationName='{0}_{1}_{5}_{4}_{2}_{3}_{6}_{7}_{8}_{9}_{10}'.format(PreTrainedClasses,ChildDatasetClasses,FPS,LearningMode,ProtocolTraining,iterationNumPrint,
    #                                                                         batchSize,datatime,ConstantBaseLearnngRate,base_lr,steps) #change this every time the implementation is changed
    ImplementationName='/{0}_{1}_{2}_{3}_TrialID_{4}_{5}_{6}_{7}_{8}'.format(PreTrainedClasses,ChildDatasetClasses,LearningMode,ProtocolTraining,ind,scedular_method,base_lr,
                                                                            batchSize,datatime) #change this every time the implementation is changed


    if not os.path.exists(root_folder):
        os.makedirs(root_folder) #creates a root folder with time sa the name
    folder_name=root_folder+ImplementationName
    logDirectory=folder_name+'/log'

    if not os.path.exists(logDirectory):
        os.makedirs(logDirectory)

    #save variables
    save_var_new(ind,config_2_static,config,config_2,ImplementationName,train_data_path,test_data_path,isStepDecay,trials,weight_path)

    class ParamMetric:
        pass

    matrixMatplot=ParamMetric()#for plotting and saving
    matrixMatplot.train_epoch_acc=[]
    matrixMatplot.test_epoch_acc=[]
    matrixMatplot.cross_entrophy_loss=[]
    matrixMatplot.cross_entrophy_loss_2=[]
    matrixMatplot.cross_entrophy_loss_3=[]
    matrixMatplot.lr=[]

    # predict_val=None
    # label_val=None
    # iteration=None

    iteration=np.zeros((ChildDatasetClasses,1))
    #iteration_final=np.zeros((class_num,1))
    predict_val=[]
    label_val=[]


    iteration_2=np.zeros((ChildDatasetClasses,1))
    #iteration_final=np.zeros((class_num,1))
    predict_val_2=[]
    label_val_2=[]


    @tf.function
    def test_step(features,labels,loss_fn='softmax_cross',label_smooth=0.01):
        __ , __, inter_logits = model(features, training=False)
        ___ , gap_embed , inter_logits_2 = model_random(inter_logits, training=False)
        logits=model_fc(inter_logits_2, training=False) #notice if we are to add drop out we have to pass training=True and change the code in model_fc

        if loss_fn=='softmax_cross':
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits,
                                                                labels=labels)
        elif loss_fn=='category_cross':
            cross_entropy=tf.keras.losses.CategoricalCrossentropy(from_logits=True,label_smoothing=label_smooth,
                                                            reduction=tf.keras.losses.Reduction.NONE)(labels,logits)
        else:
            raise NotImplementedError
        loss = tf.reduce_sum(cross_entropy) * (1.0 / global_batch_size)
        cross_entropy_loss_2(loss)
        return tf.nn.softmax(logits) , gap_embed
    
    
    @tf.function
    def test_step_FC(features,labels,loss_fn='softmax_cross',label_smooth=0.01):
        __ , __, inter_logits = model(features, training=False)
        ___ , __, inter_logits_2 = model_random(inter_logits, training=False)
        logits=model_fc(inter_logits_2, training=False) #notice if we are to add drop out we have to pass training=True and change the code in model_fc

        return tf.nn.softmax(logits) , -1



    @tf.function #comment this line to debug inside
    def train_step(features, labels,loss_fn='softmax_cross',label_smooth=0.01):
    #   def step_fn(features, labels):
        with tf.GradientTape() as tape:
            __ , _ , inter_logits = model(features, training=False) #first two are fc output and embedding for tsne ;MAY BE CHANGE FOR INFERENCE MODE!!-> training=False
            ___ , __, inter_logits_2 = model_random(inter_logits, training=True)
            logits=model_fc(inter_logits_2, training=True)

            softmax_logits=tf.keras.activations.softmax(logits)
            if loss_fn=='softmax_cross':
                cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits,
                                                                    labels=labels)
            elif loss_fn=='category_cross':
                cross_entropy=tf.keras.losses.CategoricalCrossentropy(from_logits=True,label_smoothing=label_smooth,
                                                                reduction=tf.keras.losses.Reduction.NONE)(labels,logits)
            else:
                raise NotImplementedError
            loss = tf.reduce_sum(cross_entropy) * (1.0 / global_batch_size)
        grads = tape.gradient(loss, model.trainable_variables + model_random.trainable_variables + model_fc.trainable_variables)

        optimizer.apply_gradients(list(zip(grads, model.trainable_variables + model_random.trainable_variables + model_fc.trainable_variables)))  
        train_acc(labels, logits)
        train_acc_top_5(labels, logits)
        cross_entropy_loss(loss)
        cross_entropy_loss_3(loss)

        epoch_train_acc(labels, softmax_logits)
        epoch_train_acc_top_5(labels, softmax_logits)


    @tf.function #comment this line to debug inside
    def train_step_FC(features, labels,loss_fn='softmax_cross',label_smooth=0.01):
    #   def step_fn(features, labels):
        with tf.GradientTape() as tape:
            __ , _ , inter_logits = model(features, training=True) #first two are fc output and embedding for tsne ;MAY BE CHANGE FOR INFERENCE MODE!!-> training=False
            ___ , __, inter_logits_2 = model_random(inter_logits, training=True)
            logits = model_fc(inter_logits_2, training=True)

            # softmax_logits=tf.keras.activations.softmax(logits)
            if loss_fn=='softmax_cross':
                cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits,
                                                                    labels=labels)
            elif loss_fn=='category_cross':
                cross_entropy=tf.keras.losses.CategoricalCrossentropy(from_logits=True,label_smoothing=label_smooth,
                                                                reduction=tf.keras.losses.Reduction.NONE)(labels,logits)
            else:
                raise NotImplementedError
            loss = tf.reduce_sum(cross_entropy) * (1.0 / global_batch_size)
        grads = tape.gradient(loss, model_random.trainable_variables +  model_fc.trainable_variables)

        optimizer.apply_gradients(list(zip(grads, model_random.trainable_variables +  model_fc.trainable_variables)))  



    parser = get_parser()
    arg = parser.parse_args()

    base_lr         = arg.base_lr
    num_classes     = arg.num_classes #child num classes
    epochs          = arg.num_epochs
    checkpoint_path = arg.checkpoint_path
    log_dir         = arg.log_dir
    train_data_path = train_data_path
    test_data_path  = test_data_path
    save_freq       = arg.save_freq
    steps           = arg.steps
    batch_size      = arg.batch_size
    gpus            = arg.gpus
    # strategy        = tf.distribute.MirroredStrategy(arg.gpus)#look into MirrorStrategy class
    global_batch_size = arg.batch_size   #*strategy.num_replicas_in_sync
    # arg.gpus        = strategy.num_replicas_in_sync
    iterations      = arg.iteration



    #copy hyperparameters and model definition to log folder
    save_arg(arg)
    shutil.copy2(inspect.getfile(Model), arg.log_dir)
    shutil.copy2(__file__, arg.log_dir)#save main(this file itseld) to log

    train_data = get_dataset(train_data_path,
                            num_classes=num_classes,
                            batch_size=global_batch_size,
                            drop_remainder=True,
                            shuffle=True) #notice this doesnt have all the train data but only the batch how??
    # train_data = strategy.experimental_distribute_dataset(train_data)

    test_data = get_dataset(test_data_path,
                            num_classes=num_classes,
                            batch_size=batch_size,
                            drop_remainder=False,
                            shuffle=False)# shuffle - false 

    iterationsOrSteps=iterations #40000 initially
    boundaries = [(step*iterationsOrSteps)//batch_size for step in steps]
    values = [base_lr]*(len(steps)+1)

    # scheduler selection
    if ConstantBaseLearnngRate==True:
        learning_rate  = tf.keras.optimizers.schedules.PiecewiseConstantDecay(boundaries, values,name='new')
    elif scheduler=='piecewise_decay':
        for i in range(1, len(steps)+1):
            values[i] *= 0.1**i
        learning_rate  = tf.keras.optimizers.schedules.PiecewiseConstantDecay(boundaries, values,name='new')
        # learning_rate_FC  = tf.keras.optimizers.schedules.PiecewiseConstantDecay(boundaries, values)
    elif scheduler=='expo_decay':
        steps_decay=iterationsOrSteps//batch_size
        learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(
                                        base_lr, decay_steps=steps_decay, decay_rate=decay_rate,staircase=isStepDecay)
    else:
        ValueError('Unknown scheduler: {}'.format(scheduler))



    # optimizer selection
    if opti_mode=='SGD':
        optimizer    = tf.keras.optimizers.SGD(learning_rate=learning_rate,
                                            momentum=0.9,
                                            nesterov=True)
    elif opti_mode=='ADAM':
        optimizer  = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    else:
        ValueError('Unknown optimizer: {}'.format(optimizer))
    
    model        = Model(num_classes=PreTrainedClasses,layer_out=STGCN_OUT, initializers=initial_model) #pre-trained model
    # model        = Model(num_classes=PreTrainedClasses, layer_out=STGCN_OUT) #to check without intializer object
    # ckpt         = tf.train.Checkpoint(model=model, optimizer=optimizer)
    # ckpt_manager = tf.train.CheckpointManager(ckpt,
    #                                             checkpoint_path,
    #                                             max_to_keep=20)

    # checkpoint_list=ckpt_manager.checkpoints
    # checkpoint_last=ckpt_manager.latest_checkpoint
    # status = ckpt.restore(checkpoint_last)    #.expect_partial().assert_existing_objects_matched() #checkpoint_list[0] or checkpoint_last
    # model.load_weights(checkpoint_last)
    
    model.load_weights(weight_path)

    model_random = Model_randomzied(num_classes=-1 , layer_out=STGCN_OUT,  initializers=initial_model)
    
    
    model_fc = Model2_single_layer(num_classes=num_classes,initializers=initial_model) #child model
    # model_fc = Model2_new(num_classes=num_classes, initializers=initial_model) #child model
    # model_fc = Model2(num_classes=num_classes) #to check without intializer object

    # keras metrics to hold accuracies and loss
    cross_entropy_loss   = tf.keras.metrics.Mean(name='cross_entropy_loss')
    cross_entropy_loss_3   = tf.keras.metrics.Mean(name='cross_entropy_loss_epoch') #for epoch wise?
    train_acc            = tf.keras.metrics.CategoricalAccuracy(name='train_acc')
    train_acc_top_5      = tf.keras.metrics.TopKCategoricalAccuracy(name='train_acc_top_5', k=5)
    epoch_train_acc       = tf.keras.metrics.CategoricalAccuracy(name='epoch_train_acc')
    epoch_train_acc_top_5 = tf.keras.metrics.TopKCategoricalAccuracy(name='epoch_train_acc_top_5',k=5)

    cross_entropy_loss_2   = tf.keras.metrics.Mean(name='cross_entropy_loss_2')
    epoch_test_acc       = tf.keras.metrics.CategoricalAccuracy(name='epoch_test_acc')
    epoch_test_acc_top_5 = tf.keras.metrics.TopKCategoricalAccuracy(name='epoch_test_acc_top_5')
    test_acc_top_5       = tf.keras.metrics.TopKCategoricalAccuracy(name='test_acc_top_5')
    test_acc             = tf.keras.metrics.CategoricalAccuracy(name='test_acc')
    summary_writer       = tf.summary.create_file_writer(log_dir)
    
    #get distribution of datasets
    dicTrain,dicTest,dicPrec,overallStr,trainNorm,testNorm=util.getTrainTestDistribution(train_data,test_data,batch_size,ChildDatasetClasses)
    distriStr='Train Distribution: '+str(dicTrain)+'\n'+'Test Distribution: '+str(dicTest)+'\n\n'+'Train-normalized Distribution: '+str(trainNorm)+'\n'+'Test-normalized Distribution: '+str(testNorm)+'\n\n'+'Test/(Train+Test) Precentage Distribution: '+str(dicPrec)+'\n\n'+overallStr
    util.WriteFile(distriStr,logDirectory,'datasetDistibution')

    # Get 1 batch from train dataset to get graph trace of train and test functions
    for data in test_data:
        features, labels = data
        break

    # for layer in model.layers:
    #     layer.trainable=False #all Model layers are frozen atm - since FX doesn't need any of the main model to have trainable ones

    # visualization_skeleton(features)
    




    #start first training the FC only-----------------------------------

    # model_fc.trainable=True

    FC_first_train_true=False

    if FC_first_train_true==True:   

        # Get 1 batch from train dataset to get graph trace of train and test functions
        for data in test_data:
            features, labels = data
            break

        for layer in model.layers:
            layer.trainable=False #all Model layers are frozen atm - since FX doesn't need any of the main model to have trainable ones


        train_iter_FC = 0
        test_iter_FC = 0
        epochs_FC = epochs_fc
        for epoch in range(epochs_FC):
            print("Epoch: {}".format(epoch+1))
            print("Training: ")

            for features, labels in tqdm(train_data):
                train_step_FC(features, labels, loss_fn=loss_fuction,label_smooth=label_smooth_val)
                # train_acc_top_5.reset_states()
                train_iter_FC += 1

            print("Testing: ")
            for features, labels in tqdm(test_data):
                y_pred, __ = test_step_FC(features,labels, loss_fn=loss_fuction,label_smooth=label_smooth_val)
                test_iter_FC += 1 #keep this line


    #start fine-tuning the whole model-----------------------------------
    for data in test_data:
        features, labels = data
        break

    #visualization_skeleton(features)

    print("\nST-GCN layer names:")
    for i in range(13):
        print('layer index:',i,' layer name:',model.layers[i].name)
    print("\nend ST-GCN layer names----------------------------")

        
    for layer in model.layers:
        layer.trainable=False

    if unFreezeFirst==True:
        for i in range(1,STGCN_OUT+1): #why batchnorm not unfrozen??
            print(i,model.layers[i].name)
            model.layers[i].trainable=True
                                    #from stgcn to stgcn_9 when range(1,11)
                                    # for i in range(1,10): #from stgcn to stgcn_8 when range(1,10)
            
    print("----------------Layers of the Model_randomized() - Second ST-GCN model----------------------------")
    for layer in model_random.layers:
        print(layer.name,layer )
        layer.trainable=True #since we have only upto GAP layer

    # S=unfreeze_start # default 10
    # M=unfreeze_end   # default 11

    # print("\nLAYER UNFREEZING STARTS NOW............................")
    # for i in range(S,M):
    #     print('layer index: ',i,' unfrozen layer name: ',model.layers[i].name)
    #     model.layers[i].trainable=True
    # print("FOR ALL IMPLEMENTATIONS GAP AND CONV2D LAYERS SHOULD BE FROZEN ALL THE TIME!!!!!!!!\n")

    #comment if batch norm is frozen ; if unfrozen then manually make trainable =True in train step and train step FC!!!!!!!!!
    # model.layers[0].trainable=True #batch norm layer



    # add graph of train and test functions to tensorboard graphs
    # Note:
    # graph training is True on purpose, allows tensorflow to get all the
    # variables, which is required for the first call of @tf.function function
    
    tf.summary.trace_on(graph=True)
    train_step(features, labels, loss_fn=loss_fuction,label_smooth=label_smooth_val)
    with summary_writer.as_default():
        tf.summary.trace_export(name="training_trace",step=0)
    tf.summary.trace_off()
    
    tf.summary.trace_on(graph=True)
    test_step(features,labels,loss_fn=loss_fuction,label_smooth=label_smooth_val)
    with summary_writer.as_default():
        tf.summary.trace_export(name="testing_trace", step=0)
    tf.summary.trace_off()
    # model.summary()       
    
    
    from contextlib import redirect_stdout
    with open(os.path.join(logDirectory,'modelsummary.txt'), 'w') as f:
        with redirect_stdout(f):
            model.summary()

    with open(os.path.join(logDirectory,'modelsummary_2.txt'), 'w') as f:
        with redirect_stdout(f):
            model_fc.summary()

    cross_entropy_loss_2.reset_states()
    cross_entropy_loss_3.reset_states()
    
    
    # start training
    train_iter = 0
    test_iter = 0

    #variable for CSV saving
    classLabelCSV=[str(index)+'_class_probability' for index in range(labels.shape[1])]
    csvHeader=['uniqueID','real_label','real_probability','predicted_label','predicted_probability']
    if save_all_data:
        csvHeader+=classLabelCSV
    pathCSV=Path(folder_name+'/save_values_csv/')
    if not (pathCSV.exists()):
        os.mkdir(pathCSV)

    for epoch in range(epochs):
        #save data file
        csvData=[]
        
        print("Epoch: {}".format(epoch+1))
        print("Training: ")

        for features, labels in tqdm(train_data):
            # features =util.randomSelection(features,random_choose=True,window_size=150,random_move=True) #(data, random_choose=False,window_size=-1,random_move=False)
            # # features =util.randomSelection(features,select_third=True) #used only to select one out of 3 frames i.e., 100 frames per video
            # features=util.skeletonSelection(features,structure=skeleton_structure)
            train_step(features, labels,loss_fn=loss_fuction,label_smooth=label_smooth_val)
            lr = optimizer._decayed_lr(tf.float32)
            matrixMatplot.lr.append(lr.numpy())
            print('learning r: ', lr.numpy())

            matrixMatplot.cross_entrophy_loss.append(cross_entropy_loss.result().numpy()) #don't need this in plotting
            with summary_writer.as_default():
                tf.summary.scalar("cross_entropy_loss",
                                    cross_entropy_loss.result(),
                                    step=train_iter)
                tf.summary.scalar("train_acc",
                                    train_acc.result(),
                                    step=train_iter)
                tf.summary.scalar("train_acc_top_5",
                                    train_acc_top_5.result(),
                                    step=train_iter)
            cross_entropy_loss.reset_states()
            train_acc.reset_states()
            train_acc_top_5.reset_states()
            train_iter += 1
            
        matrixMatplot.cross_entrophy_loss_3.append(cross_entropy_loss_3.result().numpy())
        matrixMatplot.train_epoch_acc.append(epoch_train_acc.result().numpy())
        with summary_writer.as_default():
            tf.summary.scalar("epoch_train_acc",
                            epoch_train_acc.result(),
                            step=epoch)
            tf.summary.scalar("epoch_train_acc_top_5",
                            epoch_train_acc_top_5.result(),
                            step=epoch)
            tf.summary.scalar("epoch_train_cross_entropy_loss",
                            cross_entropy_loss_3.result(),
                            step=epoch)                   # newly added scalar
        tmp_1=epoch_train_acc.result().numpy()  #will be saved in accuracy.txt file
        epoch_train_acc.reset_states()
        epoch_train_acc_top_5.reset_states()
        cross_entropy_loss_3.reset_states()
          

        print("Testing: ")
        embedding_list=[] #for t-sne
        labels_list=[] #for t-sne
        for features, labels in tqdm(test_data):
            y_pred ,gap_embed = test_step(features,labels,loss_fn=loss_fuction,label_smooth=label_smooth_val)
            predict_list_2,label_list_2,iteration_list_2=confusion_matrix_data_2(y_pred,labels,test_iter)

            # start = timeit.default_timer()
            if (epoch + 1) % save_freq == 0:
                save_values_csv(y_pred,labels,features,epoch)
                embedding_list.append(gap_embed.numpy().squeeze(axis=(2,3))) #squeeze is to remove extra dims in 2 and 3 axis
                                                                            #  [4,256,1,1] -> [4,256]
                labels_list.append(labels.numpy())
            # stop = timeit.default_timer()
            

            test_acc(labels, y_pred)
            epoch_test_acc(labels, y_pred)
            test_acc_top_5(labels, y_pred)
            epoch_test_acc_top_5(labels, y_pred)
            with summary_writer.as_default():
                tf.summary.scalar("test_acc",
                                test_acc.result(),
                                step=test_iter)
                tf.summary.scalar("test_acc_top_5",
                                test_acc_top_5.result(),
                                step=test_iter)
            test_acc.reset_states()
            test_acc_top_5.reset_states()
            test_iter += 1
        matrixMatplot.cross_entrophy_loss_2.append(cross_entropy_loss_2.result().numpy())
        matrixMatplot.test_epoch_acc.append(epoch_test_acc.result().numpy())
        with summary_writer.as_default():
            tf.summary.scalar("epoch_test_acc",
                            epoch_test_acc.result(),
                            step=epoch)
            tf.summary.scalar("epoch_test_acc_top_5",
                            epoch_test_acc_top_5.result(),
                            step=epoch)
            tf.summary.scalar("epoch_test_cross_entropy_loss",
                            cross_entropy_loss_2.result(),
                            step=epoch)
        
        # save accuracies
        tmp=epoch_test_acc.result().numpy() 
        util.WriteFile(f'Test accuracy for epoch {epoch}: '+str(tmp)+'------------------------------------------------------------'+
                                            f'Train accuracy for epoch {epoch}: '+str(tmp_1) + '\n', logDirectory , 'accuracies')

        epoch_test_acc.reset_states()
        epoch_test_acc_top_5.reset_states()
        cross_entropy_loss_2.reset_states()

        confusion_matrix_draw_2(predict_list_2,label_list_2,iteration_list_2,epoch)
        iteration_2=np.zeros((ChildDatasetClasses,1))
        predict_val_2=[]
        label_val_2=[]

        # start = timeit.default_timer()
        
        if (epoch + 1) % save_freq == 0: #saves the data to a  csv file
            with open(os.path.join(pathCSV,'data_{}.csv'.format(epoch+1)), 'w+', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(csvHeader)
                writer.writerows(csvData)
            pathCSV_read=os.path.join(pathCSV,'data_{}.csv'.format(epoch+1))
            df=pd.read_csv(pathCSV_read)  
            plotBoxWhisker(df,epoch)
            reliabilityDigram_2(df, num_classes, epoch, bin_n=10)
            reliabilityDigram(df, num_classes, epoch, bin_n=10) #sanity check
            tsne_out,labels_out=tsne_create(embedding_list,labels_list)

        # stop = timeit.default_timer()
        # print('Time for csv file saving: ', stop - start) 


        # if (epoch + 1) % save_freq == 0:
        #     ckpt_save_path = ckpt_manager.save()
        #     print('Saving checkpoint for epoch {} at {}'.format(epoch+1,
        #                                                         ckpt_save_path))
        if (epoch + 1) % save_freq == 0:
            plotAccuracy(matrixMatplot.train_epoch_acc,matrixMatplot.test_epoch_acc,matrixMatplot.cross_entrophy_loss_3,matrixMatplot.cross_entrophy_loss_2,epoch+1,train_iter,matrixMatplot.lr,tsne_out,labels_out,False)

    # ckpt_save_path = ckpt_manager.save()
    # print('Saving final checkpoint for epoch {} at {}'.format(epochs,
    #                                                           ckpt_save_path))

    #call matplotlib here to save the final plots for later use
    plotAccuracy(matrixMatplot.train_epoch_acc,matrixMatplot.test_epoch_acc,matrixMatplot.cross_entrophy_loss_3,matrixMatplot.cross_entrophy_loss_2,epochs,train_iter,matrixMatplot.lr,tsne_out,labels_out,True)


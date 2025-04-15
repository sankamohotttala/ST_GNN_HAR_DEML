#based on main_test.py code
#only to be used with LOOCV protocol but i might use this with later K-fold CV
#data loader is changed to load data with not  (feature,label) but  (feature,label,subjectID) so we can do the subject selection
#path for the changed tfrecord code :F:\Codes\joint attention\2022\visualize_vattu_child\tf_record_add_sampleName.py

from model.stgcn_cwbg import Model
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

from sklearn.manifold import TSNE
from matplotlib.colors import ListedColormap

#-----------utility functions------------------


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
    plt.figure(figsize=(10, 8)) #(60, 48) changed to 10,8
    sns.heatmap(confusion_mtx,xticklabels=classList,yticklabels=classList,  annot=True, fmt='.2f')
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
    print(tsne_embedded.shape)
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
    plt.figure(figsize=(10,10))
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

    plt.close('all')


def get_LOOCV_dataset(train_data_tmp,child):

    feature_tensor_list=[]
    label_tensor_list=[]
    name_tensor_list=[]

    for features, labels, name in train_data_tmp:
        feature_tensor_list.append(features)
        label_tensor_list.append(labels)
        name_tensor_list.append(name)

    feature_tensor_f=tf.concat(feature_tensor_list,axis=0)
    label_tensor_f=tf.concat(label_tensor_list,axis=0)
    name_tensor_f=tf.concat(name_tensor_list,axis=0)
    
    test_child_indices=[]
    train_child_indices=[]
    for i,name in enumerate(name_tensor_f):
        name_str=name.numpy().decode('utf-8')
        child_id=int(name_str.split('P')[1].split('R')[0])
        # print(child_id)
        if (child_id-1) ==child:  #child id starts from 1 so subtract 1 ; child starts from 0
            test_child_indices.append(i)
        else:
            train_child_indices.append(i)

    #train
    train_feature=tf.gather(feature_tensor_f,indices=train_child_indices)
    train_label=tf.gather(label_tensor_f,indices=train_child_indices)
    train_name=tf.gather(name_tensor_f,indices=train_child_indices)

    #test
    test_feature=tf.gather(feature_tensor_f,indices=test_child_indices)
    test_label=tf.gather(label_tensor_f,indices=test_child_indices)
    test_name=tf.gather(name_tensor_f,indices=test_child_indices)

    train_dataset=tf.data.Dataset.from_tensor_slices((train_feature,train_label,train_name))
    test_dataset=tf.data.Dataset.from_tensor_slices((test_feature,test_label,test_name))

    train_dataset = train_dataset.batch(4, drop_remainder=True)
    train_dataset = train_dataset.prefetch(4)
    train_dataset = train_dataset.shuffle(1000)

    test_dataset = test_dataset.batch(4, drop_remainder=False)
    test_dataset = test_dataset.prefetch(4)
    # test_dataset = test_dataset.shuffle(1000)
    return train_dataset, test_dataset

def get_dataset(directory, num_classes=60, batch_size=32, drop_remainder=False,
                shuffle=False, shuffle_size=1000): #previosuly num_classes was 60;i chanegd to fiund the reason for error
    # dictionary describing the features.
    feature_description = {
        'features': tf.io.FixedLenFeature([], tf.string),
        'label'     : tf.io.FixedLenFeature([], tf.int64),
        'name'  : tf.io.FixedLenFeature([], tf.string)
    }

    # parse each proto and, the features within
    def _parse_feature_function(example_proto):
        features = tf.io.parse_single_example(example_proto, feature_description)
        data =  tf.io.parse_tensor(features['features'], tf.float32)
        label = tf.one_hot(features['label'], num_classes)
        name = features['name']
        data = tf.reshape(data, (3, 300, 20, 1)) #changed both num_joints from 25 to 20 and num_skeletons from 2 to 1
        return data, label ,name

    records = [os.path.join(directory, file) for file in os.listdir(directory) if file.endswith("tfrecord")]
    dataset = tf.data.TFRecordDataset(records, num_parallel_reads=len(records))
    dataset = dataset.map(_parse_feature_function)
    dataset = dataset.batch(batch_size, drop_remainder=drop_remainder)
    dataset = dataset.prefetch(batch_size)
    if shuffle:
        dataset = dataset.shuffle(shuffle_size)
    return dataset


#common parameter initialization





save_all_data=True
skeleton_structure='coco' #need to probably change the graph as well

num_children=30
for child in range(num_children): #child from 0 to 29

    scheduler='expo_decay' #piecewise_decay or expo_decay
    decay_rate=0.96 #.96 default ,.5 ->e-9 value, .75 -> 0.00017..
    isStepDecay=True #if true step decay is used instead of constinous decay
    ConstantBaseLearnngRate=False #if true then base_lr will be same ;if False LRschedular will be used as in the code
    base_lr=1e-2 #0.01 , 0.1 when epochs are 50
    steps=[10,20] # [20,30] when epochs are 50
    # steps=[1,5]

    PreTrainedClasses=0 #since no transfer learning here
    ChildDatasetClasses=5 #change the train, val data paths as well
    ProtocolTraining='loocv_shared' #since there is only one
    batchSize=4

    LearningMode=f'person_{child+1}'  #without_strategy_testCross
    FPS=30 #either 10 or 30
    #checkpoint_restore_path=None        #this is given a value below
    iterationNum=432 #cwbg all -  , similar - , dissimilar - 868, shared -432
    if ConstantBaseLearnngRate==True:
        iterationNumPrint= -1   #since lr is constant
    else:
        iterationNumPrint=iterationNum

    used_epochs = 30 # original 50
    # for CFBG dataset
    if ProtocolTraining=='loocv_diss':    
        train_data_path = r'F:\Codes\joint attention\2022\visualize_vattu_child\new_protocols\for_LOOCV\dissimilar\experiment\train_data'
        test_data_path  = r'F:\Codes\joint attention\2022\visualize_vattu_child\new_protocols\for_LOOCV\dissimilar\experiment\val_data'
    elif ProtocolTraining=='loocv_new':
        train_data_path = r'F:\Codes\joint attention\2022\visualize_vattu_child\new_protocols\for_LOOCV\full\experiment\train_data' 
        test_data_path  = r'F:\Codes\joint attention\2022\visualize_vattu_child\new_protocols\for_LOOCV\full\experiment\val_data'
    elif ProtocolTraining=='loocv_shared':
        train_data_path=r'F:\Codes\joint attention\2022\visualize_vattu_child\new_protocols\for_LOOCV\shared\xsub\train_data'
        test_data_path=r'F:\Codes\joint attention\2022\visualize_vattu_child\new_protocols\for_LOOCV\shared\xsub\val_data'
    else:
        ValueError()

    datatime=util.data_time_string() #used as a unique number as well so we know when the code was ran


    ImplementationName='{0}_{1}_{5}_{4}_{2}_{3}_{6}_{7}_{8}_{9}_{10}'.format(PreTrainedClasses,ChildDatasetClasses,FPS,LearningMode,ProtocolTraining,iterationNumPrint,
                                                                    batchSize,datatime,ConstantBaseLearnngRate,base_lr,steps) #change this every time the implementation is changed
    folder_name='results_cwbg_new_protocol/loocv_sim_1/'+ImplementationName #root folder also gets created 
    logDirectory=folder_name+'/log'
    checkpoint_restore_path=os.path.join(r'F:\Codes\joint attention\2022 - Journal\zST-GCN_CWBG_LOOP\checkpoints\loocv_sim_1',ImplementationName)


    class ParamMetric:
        pass

    matrixMatplot=ParamMetric()    #for plotting and saving
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


#-----------------

    @tf.function
    def test_step(features,labels):
        logits , gap_embed = model(features, training=False)
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits,
                                                                labels=labels)
        loss = tf.reduce_sum(cross_entropy) * (1.0 / global_batch_size)
        cross_entropy_loss_2(loss)
        # loss_test.append(loss) #for loss histogram on tensorboard
        return tf.nn.softmax(logits) , gap_embed


    @tf.function #comment this line to debug inside
    def train_step(features, labels):
    #   def step_fn(features, labels):
        with tf.GradientTape() as tape:
            logits , __ = model(features, training=True)

            softmax_logits=tf.keras.activations.softmax(logits)
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits,
                                                                    labels=labels)
            loss = tf.reduce_sum(cross_entropy) * (1.0 / global_batch_size)
        grads = tape.gradient(loss, model.trainable_variables)

        optimizer.apply_gradients(list(zip(grads, model.trainable_variables)))  
        train_acc(labels, logits)
        train_acc_top_5(labels, logits)
        cross_entropy_loss(loss)
        cross_entropy_loss_3(loss)

        epoch_train_acc(labels, softmax_logits)
        epoch_train_acc_top_5(labels, softmax_logits)
        # paramss.action=softmax_logits


    parser = get_parser()
    arg = parser.parse_args()

    base_lr         = arg.base_lr
    num_classes     = arg.num_classes
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

    train_data_tmp = get_dataset(train_data_path,
                             num_classes=num_classes,
                             batch_size=global_batch_size,
                             drop_remainder=False,
                             shuffle=True) # drop remiander = False sinc ewe need all data for LOOCV
    
    #create a dataset with all the data
    train_LOOCV,test_LOOCV=get_LOOCV_dataset(train_data_tmp,child)

    for features, labels,name in train_LOOCV:
        for i in range(len(name)):
            # print(name.numpy()[i].decode('utf-8'))
            pass
    
    # print('test_dataset')
    for features, labels,name in test_LOOCV:
        for i in range(len(name)):
            # print(name.numpy()[i].decode('utf-8'))
            pass


    #replace train_data with train_LOOCV
    train_data = train_LOOCV
    
    #replace test_data with test_LOOCV
    test_data = test_LOOCV

    iterationsOrSteps=iterations #40000 initially
    boundaries = [(step*iterationsOrSteps)//batch_size for step in steps]
    values = [base_lr]*(len(steps)+1)

    
    # scheduler selection
    if scheduler=='piecewise_decay':
        if ConstantBaseLearnngRate==False:
            for i in range(1, len(steps)+1):
                values[i] *= 0.1**i
        learning_rate  = tf.keras.optimizers.schedules.PiecewiseConstantDecay(boundaries, values)
        # learning_rate_FC  = tf.keras.optimizers.schedules.PiecewiseConstantDecay(boundaries, values)
    elif scheduler=='expo_decay':
        steps_decay=iterationsOrSteps//batch_size
        learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(
                                        base_lr, decay_steps=steps_decay, decay_rate=decay_rate,staircase=isStepDecay)
    else:
        ValueError('Unknown scheduler: {}'.format(scheduler))

    model        = Model(num_classes)
    optimizer    = tf.keras.optimizers.SGD(learning_rate=learning_rate,
                                            momentum=0.9,
                                            nesterov=True)
    ckpt         = tf.train.Checkpoint(model=model, optimizer=optimizer)
    ckpt_manager = tf.train.CheckpointManager(ckpt,
                                                checkpoint_path,
                                                max_to_keep=20)

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
        features, labels, _ = data
        break
    
    # visualization_skeleton(features)
    
    # add graph of train and test functions to tensorboard graphs
    # Note:
    # graph training is True on purpose, allows tensorflow to get all the
    # variables, which is required for the first call of @tf.function function
    # loss_test=[]
    tf.summary.trace_on(graph=True)
    train_step(features, labels)
    with summary_writer.as_default():
      tf.summary.trace_export(name="training_trace",step=0)
    tf.summary.trace_off()
    
    tf.summary.trace_on(graph=True)
    test_step(features,labels)
    with summary_writer.as_default():
      tf.summary.trace_export(name="testing_trace", step=0)
    tf.summary.trace_off()
    model.summary(print_fn=lambda x: util.WriteFile(x,logDirectory,'modelSummary'))
    model.summary()

    cross_entropy_loss_2.reset_states()
    cross_entropy_loss_3.reset_states()
    cross_entropy_loss.reset_states() #resets iteration wise loss
    
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
        loss_test=[]
        #save data file
        csvData=[]
        
        print("Epoch: {}".format(epoch+1))
        print("Training: ")

        for features, labels ,_ in tqdm(train_data):
            # features =util.randomSelection(features,random_choose=True,window_size=150,random_move=True) #(data, random_choose=False,window_size=-1,random_move=False)
            # # features =util.randomSelection(features,select_third=True) #used only to select one out of 3 frames i.e., 100 frames per video
            # features=util.skeletonSelection(features,structure=skeleton_structure)
            train_step(features, labels)
            lr = optimizer._decayed_lr(tf.float32)
            matrixMatplot.lr.append(lr.numpy())

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
        epoch_train_acc.reset_states()
        epoch_train_acc_top_5.reset_states()
        cross_entropy_loss_3.reset_states()  

        print("Testing: ")
        embedding_list=[] #for t-sne
        labels_list=[] #for t-sne

        start = timeit.default_timer()
        num_batchs=0
        for features, labels, _ in tqdm(test_data):
            y_pred ,gap_embed = test_step(features,labels)
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
            num_batchs+=1
        stop = timeit.default_timer()

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
            tf.summary.histogram("histogram_test_loss",
                loss_test,
                step=epoch)
        
        # save accuracies
        tmp=epoch_test_acc.result().numpy() 
        util.WriteFile(f'Test accuracy for epoch {epoch}: '+str(tmp)+'\n',logDirectory,'accuracies')

        time_inf =stop - start
        sample_inf = time_inf/(num_batchs*batch_size)
        util.WriteFile(f'Test time for epoch {epoch}: '+str(time_inf)+' , test time for sample: '+str(sample_inf)+'\n',logDirectory,'time')

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
                tsne_out,labels_out=tsne_create(embedding_list,labels_list)
        # stop = timeit.default_timer()
        # print('Time for csv file saving: ', stop - start) 

        if (epoch + 1) % save_freq == 0:
            ckpt_save_path = ckpt_manager.save()
            print('Saving checkpoint for epoch {} at {}'.format(epoch+1,
                                                                ckpt_save_path))
        if (epoch + 1) % save_freq == 0:
            plotAccuracy(matrixMatplot.train_epoch_acc,matrixMatplot.test_epoch_acc,matrixMatplot.cross_entrophy_loss_3,matrixMatplot.cross_entrophy_loss_2,epoch+1,train_iter,matrixMatplot.lr,tsne_out,labels_out,False)

    ckpt_save_path = ckpt_manager.save()
    print('Saving final checkpoint for epoch {} at {}'.format(epochs,
                                                              ckpt_save_path))

    #call matplotlib here to save the final plots for later use
    plotAccuracy(matrixMatplot.train_epoch_acc,matrixMatplot.test_epoch_acc,matrixMatplot.cross_entrophy_loss_3,matrixMatplot.cross_entrophy_loss_2,epochs,train_iter,matrixMatplot.lr,tsne_out,labels_out,True)


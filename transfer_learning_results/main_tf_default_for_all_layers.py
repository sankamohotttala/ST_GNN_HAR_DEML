#done for the plot

from re import M
from model.stgcn_default_changed import Model, Model2, Model2_single_layer
import utilities as util 
import tensorflow as tf
from tqdm import tqdm
import argparse
import inspect
import shutil
import yaml
import os
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from pathlib import Path

ConstantBaseLearnngRate=False #if true then base_lr will be same ;if False LRschedular will be used as in the code
PreTrainedClasses=44 #change the checkpoint folder as well
ChildDatasetClasses=8 #change the train, val data paths as well
ProtocolTraining='S5' #S1 - Full dataset, S2 - Random dataset and S3 - Dissimilar Dataset
batchSize=4

datatime=util.data_time_string() #used as a unique number as well so we know when the code was ran
LearningMode='TFL-F' #  TFL-F for "frozen layer transfer learning", TFL-P for "TFL woth propagation method"     
                    #and FX for "feature extraction" and CL for "curriculum learning" and FS for "from sratch" 
FPS=30 #either 10 or 30
checkpoint_restore_path="checkpoints/STGCN_44"
iterationNum=1000  #628  
if ConstantBaseLearnngRate==True:
    iterationNumPrint= -1   #since lr is constant
else:
    iterationNumPrint=iterationNum

# for CFBG dataset
if ProtocolTraining =='S2':    
    train_data_path = r'F:\Codes\joint attention\2022\visualize_vattu_child\save_directory_2\xsub\train_data' #there are for random 10 classes
    test_data_path  = r'F:\Codes\joint attention\2022\visualize_vattu_child\save_directory_2\xsub\val_data'
elif ProtocolTraining =='S3':
    train_data_path = r'F:\Codes\joint attention\2022\visualize_vattu_child\save_directory_3\xsub\train_data'  #these train val are for the different 10 classes
    test_data_path  = r'F:\Codes\joint attention\2022\visualize_vattu_child\save_directory_3\xsub\val_data'
elif ProtocolTraining =='S1':
    train_data_path = r'F:\Codes\joint attention\2022\visualize_vattu_child\save_directory\xsub\train_data'  #these train val are for the full 15 classes
    test_data_path  = r'F:\Codes\joint attention\2022\visualize_vattu_child\save_directory\xsub\val_data'
elif ProtocolTraining =='S4':
    train_data_path = r'F:\Codes\joint attention\2022\visualize_vattu_child\save_directory_4\xsub\train_data'  #these train val are for the full 15 classes
    test_data_path  = r'F:\Codes\joint attention\2022\visualize_vattu_child\save_directory_4\xsub\val_data'
elif ProtocolTraining =='S5':
    train_data_path = r'F:\Codes\joint attention\2022\visualize_vattu_child\save_directory_Comparison\xsub\train_data'
    test_data_path  = r'F:\Codes\joint attention\2022\visualize_vattu_child\save_directory_Comparison\xsub\val_data'
else:
    ValueError()


ImplementationName='{0}_propagation-layer-plot_{1}_class_{5}_iterations_protocol_{4}_{2}FPS_{3}_note_{6}batchsize{7}'.format(PreTrainedClasses,ChildDatasetClasses,
                                                                                    FPS,LearningMode,ProtocolTraining,iterationNumPrint,batchSize,datatime) #change this every time the implementation is changed
#NumClass=44 #used in conf matrix

folder_name='images/'+ImplementationName
logDirectory=folder_name+'/log'


loss_sav=1
class Paramss:    
    pass
paramss=Paramss()
paramss.action=None

class ParamMetric:
    pass

matrixMatplot=ParamMetric()#for plotting and saving
matrixMatplot.train_epoch_acc=[]
matrixMatplot.test_epoch_acc=[]
matrixMatplot.cross_entrophy_loss=[]
matrixMatplot.cross_entrophy_loss_2=[]
matrixMatplot.cross_entrophy_loss_3=[]

predict_val=None
label_val=None
iteration=None

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
        '--base-lr', type=float, default=1e-1, help='initial learning rate')
    parser.add_argument(
        '--num-classes', type=int, default=ChildDatasetClasses, help='number of classes in dataset')
    parser.add_argument(
        '--batch-size', type=int, default=batchSize, help='training batch size')
    parser.add_argument(
        '--num-epochs', type=int, default=50, help='total epochs to train')
    parser.add_argument(
        '--save-freq', type=int, default=1000, help='periodicity of saving model weights')
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
        default=[10, 30],
        nargs='+',
        help='the epoch where optimizer reduce the learning rate, eg: 10 50')#[20, 200] was the previous one used//// [10 30] used later
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


def save_arg(arg):
    # save arg
    arg_dict = vars(arg)
    if not os.path.exists(arg.log_dir):
        os.makedirs(arg.log_dir)
    with open(os.path.join(arg.log_dir, "config.yaml"), 'w') as f:
        yaml.dump(arg_dict, f)



def get_dataset(directory, num_classes=10, batch_size=32, drop_remainder=False,
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
        data = tf.reshape(data, (3, 300, 20, 1))
        return data, label

    records = [os.path.join(directory, file) for file in os.listdir(directory) if file.endswith("tfrecord")]
    dataset = tf.data.TFRecordDataset(records, num_parallel_reads=len(records))
    dataset = dataset.map(_parse_feature_function)
    dataset = dataset.batch(batch_size, drop_remainder=drop_remainder)
    dataset = dataset.prefetch(batch_size)
    if shuffle:
        dataset = dataset.shuffle(shuffle_size)
    return dataset



@tf.function
def test_step(features,labels):
    logits = model(features, training=False)
    logits=model_fc(logits, training=False)
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits,
                                                              labels=labels)
    loss = tf.reduce_sum(cross_entropy) * (1.0 / global_batch_size)
    cross_entropy_loss_2(loss)
    return tf.nn.softmax(logits)

@tf.function
def test_step_FC(features,labels):
    logits = model(features, training=False)
    logits=model_fc(logits, training=False)
    # cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits,
    #                                                           labels=labels)
    # loss = tf.reduce_sum(cross_entropy) * (1.0 / global_batch_size)
    # cross_entropy_loss_2(loss)
    return tf.nn.softmax(logits)


@tf.function #comment this line to debug inside
def train_step(paramss,features, labels):
  def step_fn(features, labels):
    with tf.GradientTape() as tape:
      logits = model(features, training=False)
      for layer in model.layers[1].layers:
        #print(layer.name, layer)
        pass
      #print(model.layers[10].weights)
      logits=model_fc(logits, training=True) #should always be true
      # print(logits)
      softmax_logits=tf.keras.activations.softmax(logits)
      cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits,
                                                              labels=labels)
      loss = tf.reduce_sum(cross_entropy) * (1.0 / global_batch_size)
      
    grads = tape.gradient(loss, model_fc.trainable_variables + model.trainable_variables) #do we have to have order (i.e., layers order) in training variable list??

    optimizer.apply_gradients(list(zip(grads, model_fc.trainable_variables + model.trainable_variables)))
    train_acc(labels, logits)
    train_acc_top_5(labels, logits)
    cross_entropy_loss(loss)
    cross_entropy_loss_3(loss)

    epoch_train_acc(labels, softmax_logits)
    epoch_train_acc_top_5(labels, softmax_logits)
    paramss.action=softmax_logits

  strategy.run(step_fn, args=(features, labels,))


@tf.function #comment this line to debug inside
def train_step_FC(paramss,features, labels):
  def step_fn(features, labels):
    with tf.GradientTape() as tape:
      logits = model(features, training=False)
      for layer in model.layers[1].layers:
        #print(layer.name, layer)
        pass
      #print(model.layers[10].weights)
      logits=model_fc(logits, training=True)
      # print(logits)
      softmax_logits=tf.keras.activations.softmax(logits)
      cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits,
                                                              labels=labels)
      loss = tf.reduce_sum(cross_entropy) * (1.0 / global_batch_size)
      
    grads = tape.gradient(loss, model_fc.trainable_variables ) 

    optimizer.apply_gradients(list(zip(grads, model_fc.trainable_variables )))

  strategy.run(step_fn, args=(features, labels,))


#------------------------------confusion matrix part----------------------
#-------------------------------------------------------------------------

iteration=np.zeros((ChildDatasetClasses,1))
#iteration_final=np.zeros((class_num,1))
predict_val=[]
label_val=[]


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


iteration_2=np.zeros((ChildDatasetClasses,1))
#iteration_final=np.zeros((class_num,1))
predict_val_2=[]
label_val_2=[]
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
    plt.figure(figsize=(10, 8))
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

def plotAccuracy(train_epoch,test_epoch,cross_loss,cross_loss_2,epochsNumber,train_iteration_final,test_iteration_final):
    epochsNum = range(epochsNumber)
    train_iteration = range(train_iteration_final)
    test_iteration = range(test_iteration_final)
    pathDirectoryCF=Path(folder_name+'/matplotlib_graphs/')
    if not (pathDirectoryCF.exists()):
        os.mkdir(pathDirectoryCF)

    plt.figure()
    plt.plot(epochsNum, train_epoch, 'k', label='Training epoch accuracy')
    plt.title('Training accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuaracy')
    #plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(pathDirectoryCF,'Train_plot.jpg'))
    plt.close()

    plt.figure()
    plt.plot(epochsNum, test_epoch, 'k', label='Test epoch accuracy')
    plt.title('Testing accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuaracy')
    #plt.legend()
    plt.grid(True)
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
    plt.grid(True)
    plt.savefig(os.path.join(pathDirectoryCF,'Train_test_plot.jpg'))
    plt.close()

    plt.figure()
    plt.plot(train_iteration, cross_loss, 'k', label='cross entropy loss')
    plt.title('Cross Entropy Loss - Training')
    plt.xlabel('epochs')
    plt.ylabel('Loss')
    #plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(pathDirectoryCF,'cross_loss_plot.jpg'))
    plt.close()
    
    plt.figure()
    plt.plot(test_iteration, cross_loss_2, 'k', label='cross entropy loss')
    plt.title('Cross Entropy Loss - Test')
    plt.xlabel('epochs')
    plt.ylabel('Loss')
    #plt.legend()
    plt.grid(True)
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
    plt.grid(True)
    plt.savefig(os.path.join(pathDirectoryCF,'Train_test_cross_entropy_plot.jpg'))
    plt.close()

if __name__ == "__main__":
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
    strategy        = tf.distribute.MirroredStrategy(arg.gpus)#look into MirrorStrategy class
    global_batch_size = arg.batch_size*strategy.num_replicas_in_sync
    arg.gpus        = strategy.num_replicas_in_sync
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
    train_data = strategy.experimental_distribute_dataset(train_data)

    test_data = get_dataset(test_data_path,
                            num_classes=num_classes,
                            batch_size=batch_size,
                            drop_remainder=False,
                            shuffle=False)# shuffle - false 
    iterationsOrSteps=iterations #40000 initially
    boundaries = [(step*iterationsOrSteps)//batch_size for step in steps]
    values = [base_lr]*(len(steps)+1)
    if ConstantBaseLearnngRate==False:
        for i in range(1, len(steps)+1):
            values[i] *= 0.1**i
    learning_rate  = tf.keras.optimizers.schedules.PiecewiseConstantDecay(boundaries, values)
    # learning_rate_FC  = tf.keras.optimizers.schedules.PiecewiseConstantDecay(boundaries, values)

    with strategy.scope():
        model        = Model(num_classes=PreTrainedClasses)
        optimizer    = tf.keras.optimizers.SGD(learning_rate=learning_rate,
                                               momentum=0.9,
                                               nesterov=True)
        # optimizer  = tf.keras.optimizers.Adam()

        ckpt         = tf.train.Checkpoint(model=model, optimizer=optimizer)
        ckpt_manager = tf.train.CheckpointManager(ckpt,
                                                  checkpoint_path,
                                                  max_to_keep=5)
        # checkpoint_list=ckpt_manager.checkpoints
        checkpoint_last=ckpt_manager.latest_checkpoint
        status = ckpt.restore(checkpoint_last)
        model_fc = Model2(num_classes=ChildDatasetClasses) #multiple layers
        # model_fc = Model2_single_layer(num_classes=ChildDatasetClasses) #single layer

        # keras metrics to hold accuracies and loss
        cross_entropy_loss   = tf.keras.metrics.Mean(name='cross_entropy_loss')
        cross_entropy_loss_3   = tf.keras.metrics.Mean(name='cross_entropy_loss_epoch')        
        train_acc            = tf.keras.metrics.CategoricalAccuracy(name='train_acc')
        train_acc_top_5      = tf.keras.metrics.TopKCategoricalAccuracy(name='train_acc_top_2',k=2)
        epoch_train_acc       = tf.keras.metrics.CategoricalAccuracy(name='epoch_train_acc')
        epoch_train_acc_top_5 = tf.keras.metrics.TopKCategoricalAccuracy(name='epoch_train_acc_top_2',k=2)

    cross_entropy_loss_2   = tf.keras.metrics.Mean(name='cross_entropy_loss_2')
    epoch_test_acc       = tf.keras.metrics.CategoricalAccuracy(name='epoch_test_acc')
    epoch_test_acc_top_5 = tf.keras.metrics.TopKCategoricalAccuracy(name='epoch_test_acc_top_2',k=2)
    test_acc_top_5       = tf.keras.metrics.TopKCategoricalAccuracy(name='test_acc_top_2',k=2)
    test_acc             = tf.keras.metrics.CategoricalAccuracy(name='test_acc')
    summary_writer       = tf.summary.create_file_writer(log_dir)
    
    #start first training the FC only-----------------------------------
    FC_first_train_true=True

    if FC_first_train_true==True:

        for data in test_data:
            features, labels = data
            break
        
        for layer in model.layers:
            layer.trainable=False #all Model layers are frozen atm

        # tf.summary.trace_on(graph=True)
        # train_step_FC(paramss,features, labels)
        # with summary_writer.as_default():
        #     tf.summary.trace_export(name="training_trace",step=0)
        # tf.summary.trace_off()
        
        # tf.summary.trace_on(graph=True)
        # test_step_FC(features,labels)
        # with summary_writer.as_default():
        #     tf.summary.trace_export(name="testing_trace", step=0)
        # tf.summary.trace_off()
        # model.summary()

        # start training
        train_iter_FC = 0
        test_iter_FC = 0
        epochs_FC=3
        for epoch in range(epochs_FC):
            print("Epoch: {}".format(epoch+1))
            print("Training: ")
            with strategy.scope():
                for features, labels in tqdm(train_data):
                    train_step_FC(paramss,features, labels)
                    # train_acc_top_5.reset_states()
                    train_iter_FC += 1

            print("Testing: ")
            for features, labels in tqdm(test_data):
                y_pred = test_step_FC(features,labels)
                test_iter_FC += 1 #keep this line
    #ends first training the FC only---------------------------------  


    #starts fine tuning the entire model
    # Get 1 batch from train dataset to get graph trace of train and test functions
    for data in test_data:
        features, labels = data
        break

    #visualization_skeleton(features)

    a=model.layers
    a=len(model.layers)
    print(model.layers[10].name)
    # for layer in model.layers[1].layers:
    #     print(layer.name, layer)
    istranable= model.layers[1].trainable
    for layer in model.layers:
        print(layer.name,layer )

    for i in range(13):
        print('layer index:',i,' layer name:',model.layers[i].name)


        
    for layer in model.layers:
        layer.trainable=False
    
    #used above instead of 'model.trainable=False'
    # N=10
    '''number of layers to unfreeze [N->(10,-1)] ; 10 for all frozen[basically FX] thus we use N-> (10,-1)
    and -1 for all unfrozen[basically propagation ]
    if N=x then upto and including x-STGCN layer are frozen
    Notice layers start rom batchnorm, stgcn ,stgcn_1, stgcn_2 etc. so stgcn start at layer number 1 not 0

    ''' 
    # for i in range(10,N,-1):
    #     model.layers[i].trainable=True
    #     print(i,model.layers[i].name)

    S=10 #layer index S is the start ofthe unfreezing thus S->[1,11] when 10 final stgcn layer unfrozen when 11 non is unrozen
        # when 1, all are unfrozen (except batch norm?? should we start from 0??)
    M=11 #thus layer index 10 is the final stgcn layer 
        #we keep the conv2d and GAP layers frozen
        #M should be a constant noh?? M=11
    print("layers to be unfrozen are......................")    
    for i in range(S,M):# from S (incluidng S) to M-1, i ranges from one val to another.
        print(i,model.layers[i].name)
    print("layers are being unfrozen.....................")
    for i in range(S,M):
        # print(i,model.layers[i].name)
        model.layers[i].trainable=True
    print("FOR ALL IMPLEMENTATIONS GAP AND CONV2D LAYERS SHOULD BE FROZEN ALL THE TIME!!!!!!!!")
        
    
    # model.layers[9].trainable=True
    # model.layers[8].trainable=True

    # assert model.trainable == False  # All layers in `model` are now frozen
    # assert model.layers[10].trainable == True  # only final stgcn layer should be trainable



    # for data_t in train_data:
    #     features_t, labels_t = data_t
    #     break

    # add graph of train and test functions to tensorboard graphs
    # Note:
    # graph training is True on purpose, allows tensorflow to get all the
    # variables, which is required for the first call of @tf.function function
    
    tf.summary.trace_on(graph=True)
    train_step(paramss,features, labels)
    with summary_writer.as_default():
      tf.summary.trace_export(name="training_trace",step=0)
    tf.summary.trace_off()
    
    tf.summary.trace_on(graph=True)
    test_step(features,labels)
    with summary_writer.as_default():
      tf.summary.trace_export(name="testing_trace", step=0)
    tf.summary.trace_off()
    model.summary()

    #tf.keras.utils.plot_model(model, "my_first_model_with_shape_info.png", show_shapes=True)


    
    # start training
    train_iter = 0
    test_iter = 0
    for epoch in range(epochs):
        print("Epoch: {}".format(epoch+1))
        print("Training: ")
        with strategy.scope():
            for features, labels in tqdm(train_data):
                train_step(paramss,features, labels)
                predict_softmax_logit=paramss.action
                #print(paramss.action)
                #print(paramss.action.shape)

                #epoch_train_acc(labels, predict_softmax_logit)
                #epoch_train_acc_top_5(labels, predict_softmax_logit)
                matrixMatplot.cross_entrophy_loss.append(cross_entropy_loss.result().numpy()) #don't need this in plotting
                with summary_writer.as_default():
                    tf.summary.scalar("cross_entropy_loss",
                                      cross_entropy_loss.result(),
                                      step=train_iter)
                    tf.summary.scalar("train_acc",
                                      train_acc.result(),
                                      step=train_iter)
                    tf.summary.scalar("train_acc_top_2",
                                      train_acc_top_5.result(),
                                      step=train_iter)
                cross_entropy_loss.reset_states()
                train_acc.reset_states()
                train_acc_top_5.reset_states()
                train_iter += 1
            # _=epoch_train_acc.result()
            # print(_)
            matrixMatplot.cross_entrophy_loss_3.append(cross_entropy_loss_3.result().numpy())
            matrixMatplot.train_epoch_acc.append(epoch_train_acc.result().numpy())
            with summary_writer.as_default():
                tf.summary.scalar("epoch_train_acc",
                              epoch_train_acc.result(),
                              step=epoch)
                tf.summary.scalar("epoch_train_acc_top_2",
                              epoch_train_acc_top_5.result(),
                              step=epoch)
        epoch_train_acc.reset_states()
        epoch_train_acc_top_5.reset_states()
        cross_entropy_loss_3.reset_states()  

        print("Testing: ")
        for features, labels in tqdm(test_data):
            y_pred = test_step(features,labels)
            predict_list,label_list,iteration_list=confusion_matrix_data(y_pred,labels,test_iter)
            predict_list_2,label_list_2,iteration_list_2=confusion_matrix_data_2(y_pred,labels,test_iter)

            predict_val=predict_list
            label_val=label_list
            iteration=iteration_list
            
            test_acc(labels, y_pred)
            epoch_test_acc(labels, y_pred)
            test_acc_top_5(labels, y_pred)
            epoch_test_acc_top_5(labels, y_pred)
            # matrixMatplot.cross_entrophy_loss_2.append(cross_entropy_loss_2.result().numpy())
            with summary_writer.as_default():
                tf.summary.scalar("test_acc",
                                  test_acc.result(),
                                  step=test_iter)
                tf.summary.scalar("test_acc_top_2",
                                  test_acc_top_5.result(),
                                  step=test_iter)
                # tf.summary.scalar("cross_entropy_loss_2",
                #     cross_entropy_loss_2.result(),
                #                 step=test_iter)
            test_acc.reset_states()
            test_acc_top_5.reset_states()
            # cross_entropy_loss_2.reset_states()
            test_iter += 1 #keep this line
        matrixMatplot.cross_entrophy_loss_2.append(cross_entropy_loss_2.result().numpy())
        matrixMatplot.test_epoch_acc.append(epoch_test_acc.result().numpy())
        with summary_writer.as_default():
            tf.summary.scalar("epoch_test_acc",
                              epoch_test_acc.result(),
                              step=epoch)
            tf.summary.scalar("epoch_test_acc_top_2",
                              epoch_test_acc_top_5.result(),
                              step=epoch)
        epoch_test_acc.reset_states()
        epoch_test_acc_top_5.reset_states()
        cross_entropy_loss_2.reset_states()
        confusion_matrix_draw(predict_list,label_list,iteration_list,epoch)
        confusion_matrix_draw_2(predict_list_2,label_list_2,iteration_list_2,epoch)
        iteration_2=np.zeros((ChildDatasetClasses,1))
        predict_val_2=[]
        label_val_2=[]
    
    #call matplotlib here to save the final plots for later use
    plotAccuracy(matrixMatplot.train_epoch_acc,matrixMatplot.test_epoch_acc,matrixMatplot.cross_entrophy_loss_3,matrixMatplot.cross_entrophy_loss_2,epochs,epochs,epochs)

    #plt.show()
    #aa=[loss_sav]
    #     if (epoch + 1) % save_freq == 0:
    #         ckpt_save_path = ckpt_manager.save()
    #         print('Saving checkpoint for epoch {} at {}'.format(epoch+1,
    #                                                             ckpt_save_path))

    # ckpt_save_path = ckpt_manager.save()
    # print('Saving final checkpoint for epoch {} at {}'.format(epochs,
    #                                                           ckpt_save_path))


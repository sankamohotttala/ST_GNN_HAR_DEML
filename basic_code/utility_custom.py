from matplotlib import pyplot as plt
from pathlib import Path
import os
import datetime as dt
import tools
import tensorflow as tf

import numpy as np


def plotAccuracy_2(train_epoch,test_epoch,cross_loss,cross_loss_2,epochsNumber,train_iteration_final,test_iteration_final):
    global folder_name
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

# Define a useful name for our model to make it easy for us while navigating through multiple saved models.


def data_time_string():
    date_time_format = '%Y_%m_%d__%H_%M_%S'
    current_date_time_dt = dt.datetime.now()
    current_date_time_string = dt.datetime.strftime(current_date_time_dt, date_time_format)
    return current_date_time_string

def getTrainTestDistribution(trainDataset,testDataset,batchSize,classesNum):
    TrainDistribution={x:0 for x in range(classesNum)}
    TestDistribution={x:0 for x in range(classesNum)}
    trainSamples=0
    testSamples=0
    for eachBatch in trainDataset:
        label=eachBatch[1]
        batchSize=label.shape[0] #we dont use the global batch size since sometimes the final batch can have less tahn that size
                                #specially in test dataset
        for i in range(batchSize):
            testLabel=label[i]
            index=np.where(label[i] == 1)[0][0] #much easiyker i guess if converted from tf.tensor to numpy array first
            TrainDistribution[index]+=1
            trainSamples+=1
    
    for eachBatch in testDataset:
        label=eachBatch[1]
        batchSize=label.shape[0]        
        for i in range(batchSize):
            # label[i]
            index=np.where(label[i] == 1)[0][0]
            TestDistribution[index]+=1
            testSamples+=1
    
    #normalize trian and test distributions
    trainNormalizeDis={key:(value/trainSamples)*100 for key,value in TrainDistribution.items()}
    testNormalizeDis={key:(value/testSamples)*100 for key,value in TestDistribution.items()}
    
    precentageDistribution={x:(TestDistribution[x]/(TrainDistribution[x]+TestDistribution[x]))*100 for x in range(classesNum)}
    
    str1=TrainDistribution
    str2=TestDistribution
    str3=precentageDistribution
    str4='Train Samples: '+str(trainSamples)+'    Test Samples: '+str(testSamples)+'\n'
    str5=trainNormalizeDis
    str6=testNormalizeDis
    #later add the visualization code i.e., bar chart
    return str1,str2,str3,str4,str5,str6

def WriteFile(data,path,txtFilename):
    # path=r'H:\Kinetics-Dataset\child only data -part 2\child set 2\only_text'  #manually change this
    txtPath=os.path.join(path,txtFilename+'.txt')
    with open(txtPath, 'a') as txt:
        txt.write(data)

def randomSelection(data, random_choose=False,window_size=-1,random_move=False,select_third=False):
    # get data
    # data_numpy = np.array(self.data[index])
    #label = self.label[index]
    typeA=type(data)
    data_numpy=data.numpy() #TFEgerTesnor to numpy array
    size= data_numpy.shape[0]

    def forEachSample(data_numpy,random_choose,window_size,random_move):# processing
        if random_choose:
            data_numpy = tools.random_choose(data_numpy, window_size)
        elif window_size > 0:
            data_numpy = tools.auto_pading(data_numpy, window_size)
        if random_move:
            data_numpy = tools.random_move(data_numpy) #what does this exactly do?? 
        if select_third:
            data_numpy = tools.selectEachThirdFrame(data_numpy)
        return data_numpy
    tmpList=[]
    for index in range(size):
        sample=data_numpy[index]
        selected_sample=forEachSample(sample,random_choose,window_size,random_move)
        typeB=type(selected_sample)
        tmpList.append(selected_sample)
    #add elements in a list as a numpy array
    tmpNumpyt=np.array(tmpList)
    typeC=type(tmpNumpyt)
    tensorEager=tf.constant(tmpNumpyt)
    typeD=type(tensorEager)
    return tensorEager
    
'''need to chnage the method for random selection to work with kinetics camera based datasets'''
def skeletonSelection(data, structure='body_25'): # for skeleton selection
    typeA=type(data)
    data_numpy=data.numpy() #TFEgerTesnor to numpy array
    size= data_numpy.shape[0]
    coco=[True]*8+ [False]*1+ [True]*10+ [False]*6
    no_foot=[True]*19+[False]*6 # mid hip is not there

    def forEachSample(data_numpy,structure):# processing
        pass
        if structure == 'body_25':
            data_numpy=data_numpy
        if structure == 'coco':
            data_numpy=data_numpy[:,:,coco,:]
        if structure == 'no_foot':
            data_numpy=data_numpy[:,:,no_foot,:]
        return data_numpy

    tmpList=[]
    for index in range(size):
        sample=data_numpy[index]
        selected_sample=forEachSample(sample,structure)
        typeB=type(selected_sample)
        tmpList.append(selected_sample)
    #add elements in a list as a numpy array
    tmpNumpyt=np.array(tmpList)
    typeC=type(tmpNumpyt)
    tensorEager=tf.constant(tmpNumpyt)
    typeD=type(tensorEager)
    return tensorEager
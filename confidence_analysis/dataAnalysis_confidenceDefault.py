#this is to be used for comparison of confidence value  kinetics-400 and class accuracy
import pickle
import numpy as np
from matplotlib import pyplot as plt

import pandas as pd
import seaborn

# pathNumpy=r'F:\Codes\joint attention\2022 - comparison paper\kineticsSTGCN_code\result_analysis\Data\400_small\val_data.npy' #test data effect
# pathPickle=r'F:\Codes\joint attention\2022 - comparison paper\kineticsSTGCN_code\result_analysis\Data\400_small\val_label.pkl' #test data effect

pathNumpy=r'F:\Codes\joint attention\2022 - comparison paper\kineticsSTGCN_code\data\st-gcn-processed-data\data\Kinetics\kinetics-skeleton\train_data.npy' #train data effect
pathPickle=r'F:\Codes\joint attention\2022 - comparison paper\kineticsSTGCN_code\data\st-gcn-processed-data\data\Kinetics\kinetics-skeleton\train_label.pkl' #train data effect

# csvPath=r'F:\Codes\joint attention\2022 - comparison paper\kineticsSTGCN_code\result_analysis\CSV\data_30_kinetics400_small.csv'  #small set ~20,000
csvPath=r'F:\Codes\joint attention\2022 - comparison paper\kineticsSTGCN_code\result_analysis\CSV\data_20_kinetics400_all.csv' #large set ~240,000

# saveCsvPath=r'F:\Codes\joint attention\2022 - comparison paper\kineticsSTGCN_code\result_analysis\Results_save\results_acc.csv'
saveCsvPath=r'F:\Codes\joint attention\2022 - comparison paper\kineticsSTGCN_code\result_analysis\Results_save\realKinetics400_1.csv'

finalCsvPath=r'F:\Codes\joint attention\2022 - comparison paper\kineticsSTGCN_code\result_analysis\results_exp\realKinetics400_2_train.csv'

imagePath=r'F:\Codes\joint attention\2022 - comparison paper\kineticsSTGCN_code\result_analysis\Results_save\tmp1.png'

classNamePath=r'F:\Codes\joint attention\2022 - comparison paper\kineticsSTGCN_code\result_analysis\labelnamesIndex.txt'

number_class=400 
sort =True #sort the accuracy values of classes
showMyClasses=True #shows the 8 classes in red

def main():

    #new implementation
    pickleData=loadPickle(pathPickle) #19796 data points
    # numpyData=loadNumpy(pathNumpy)
    numpyData=loadNumpyAsmemoryMap(pathNumpy)

    noneNum=checkVideoNone(numpyData)  # 109 videos with no joint detected

    average_conf=calculateConfidenceAverage(numpyData,pickleData,methodUsed='default')
    average_conf_frame=calculateConfidenceAverage(numpyData,pickleData,methodUsed='frame_based')  

    df =pd.read_csv(csvPath)
    acc_dic,main_dic=classClassificatioAcc(df) 
    #sort a dictionary by value
    if sort:
        acc_dic=sorted(acc_dic.items(), key=lambda x: x[1], reverse=True)
        #convert a list of tuples to a dictionary
        acc_dic=dict(acc_dic)
    class_name_list=classNameIndexMap(acc_dic) #for the use of display in plot ; if sort-True , then this is the list for the sorted one

    # plotAccuracyGraph(acc_dic,class_name_list)
    final_average_conf=averageConfidenceRepresent(average_conf,acc_dic,dicMethod='rearrange')
    final_average_conf_frame=averageConfidenceRepresent(average_conf_frame,acc_dic,dicMethod='rearrange')

    # result_dataframe=createDatadrame_Default(final_average_conf, acc_dic, class_name_list)  #only for single one; default is used
    result_dataframe=createDatadrame_Multiple(final_average_conf,final_average_conf_frame, acc_dic, class_name_list)
    result_dataframe.to_csv(finalCsvPath,index=False)
    


def createDatadrame_Default(final_average_conf, acc_dic, class_name_list):         
    df_joint_first=pd.DataFrame.from_dict(final_average_conf, orient='index', columns=['default_confidence'])
    # df_joint_second=pd.DataFrame.from_dict(sortedAverageConfSecondPerson, orient='index',columns=colSkeletonSecond)
    #add two dataframes horizontally
    # df_joint=pd.concat([df_joint_first,df_joint_second],axis=1)
    df_joint_first_final=df_joint_first.reset_index(drop=True) #remove the index and drop it so this and others can be combined


    _=list(acc_dic.items())
    data=zip(class_name_list, list(acc_dic.values()))
    df_accuracy=pd.DataFrame(data,columns=['class_name','accuracy'])

    df_final=pd.concat([df_accuracy,df_joint_first_final],axis=1)
    return df_final

def createDatadrame_Multiple(final_average_conf,final_average_conf_frame, acc_dic, class_name_list):         
    df_joint_first=pd.DataFrame.from_dict(final_average_conf, orient='index', columns=['default_confidence'])
    df_joint_second=pd.DataFrame.from_dict(final_average_conf_frame, orient='index',columns=['frame_confidence'])
    #add two dataframes horizontally
    df_joint=pd.concat([df_joint_first,df_joint_second],axis=1)
    df_joint_first_final=df_joint.reset_index(drop=True) #remove the index and drop it so this and others can be combined


    _=list(acc_dic.items())
    data=zip(class_name_list, list(acc_dic.values()))
    df_accuracy=pd.DataFrame(data,columns=['class_name','accuracy'])

    df_final=pd.concat([df_accuracy,df_joint_first_final],axis=1)
    return df_final   




def calculateConfidenceAverage(data,pickleData,methodUsed='default'):
    numpyData=data

    #get average confidence value ---------------------------------------------------------------
    distri_dict={str(i):0 for i in range(number_class)}
    accumulated_conf_dict={str(i):0 for i in range(number_class)}
    
    # methodUsed='joint_based' #'default' or 'person_based' or 'frame based' or 'joint based'
    for i in range(numpyData.shape[0]):
        label=pickleData[1][i] #index  0-399    
        add=True
        # final_val={str(i):[0 for _ in  range(18)] for i in range(number_class)}

        if numpyData[i,2,:,:,:].sum()==0: #== to sum() entire numpyArray
            add=False
        if methodUsed=='default':
            sequence_average_conf=numpyData[i,2,:,:,:].sum()/(300*18*2)
            add=True
        if methodUsed=='person_based': #no point in using this; not informative
            if numpyData[i,2,:,:,0].sum()==0 or numpyData[i,2,:,:,1].sum()==0:
                sequence_average_conf=numpyData[i,2,:,:,:].sum()/(300*18)
            else:
                sequence_average_conf=numpyData[i,2,:,:,:].sum()/(300*18*2)
        
        if methodUsed=='joint_based':
            #go through every value of a numpy array and check if it is 0  
            data=numpyData[i,2,:,:,:]
            tmp=0
            for i in range(data.shape[0]):
                for j in range(data.shape[1]):
                    for k in range(data.shape[2]):
                        if data[i,j,k]==0:
                            tmp+=1
            if add:
                sequence_average_conf=data.sum()/(300*18*2-tmp)

        if methodUsed=='frame_based':
            data=numpyData[i,2,:,:,:]
            tmp=0
            for i in range(data.shape[0]):
                for k in range(data.shape[2]):
                    if data[i,:,k].sum()==0:
                        tmp+=1 #missing frames -- max 600
            if add:
                sequence_average_conf=data.sum()/(18*(300*2-tmp))

        if add:
            distri_dict[str(label)]+=1
            accumulated_conf_dict[str(label)]+=sequence_average_conf

    average_conf_dict={str(i):(accumulated_conf_dict[str(i)]/distri_dict[str(i)]) for i in range(number_class)}
    return average_conf_dict
    
def averageConfidenceRepresent(average_conf,acc_dic,dicMethod='rearrange'):    
    #three ways to show this dic: one default - from class 0 -399; then sort by conf value; then rearrange as in acc_dic[for multiple bar chart]
    # dicMethod='rearrange' #for the momemnt use this for visualization
    average_conf_dict=average_conf

    if dicMethod=='default':
        average_conf_dict_final=average_conf_dict
    if dicMethod=='sort':
        average_conf_dict_final=sorted(average_conf_dict.items(), key=lambda x: x[1], reverse=True)
        average_conf_dict_final=dict(average_conf_dict_final)
    if dicMethod=='rearrange':  #must be used with csv data file to visualize them in the same plot
        average_conf_dict_final={key:average_conf_dict[key] for key in acc_dic.keys()}

    return average_conf_dict_final


def SaveJointValuesTwoSkeletons(sortedAverageConfFirstPerson, sortedAverageConfSecondPerson, acc_dic, class_name_list):
    #create dataframe from a dictionary of lists
    colSkeletonFirst=['P_0_S_0', 'P_0_S_1', 'P_0_S_2', 'P_0_S_3', 'P_0_S_4', 'P_0_S_5', 'P_0_S_6', 'P_0_S_7', 'P_0_S_8', 'P_0_S_9',
                     'P_0_S_10', 'P_0_S_11', 'P_0_S_12', 'P_0_S_13', 'P_0_S_14', 'P_0_S_15', 'P_0_S_16', 'P_0_S_17']
    colSkeletonSecond=['P_1_S_0', 'P_1_S_1', 'P_1_S_2', 'P_1_S_3', 'P_1_S_4', 'P_1_S_5', 'P_1_S_6', 'P_1_S_7', 'P_1_S_8', 
                    'P_1_S_9', 'P_1_S_10', 'P_1_S_11', 'P_1_S_12', 'P_1_S_13', 'P_1_S_14', 'P_1_S_15', 'P_1_S_16', 'P_1_S_17']          
    df_joint_first=pd.DataFrame.from_dict(sortedAverageConfFirstPerson, orient='index', columns=colSkeletonFirst)
    df_joint_second=pd.DataFrame.from_dict(sortedAverageConfSecondPerson, orient='index',columns=colSkeletonSecond)
    #add two dataframes horizontally
    df_joint=pd.concat([df_joint_first,df_joint_second],axis=1)
    df_joint2=df_joint.reset_index(drop=True) #remove the index and drop it so this and others can be combined


    _=list(acc_dic.items())
    data=zip(class_name_list, list(acc_dic.values()))
    df_accuracy=pd.DataFrame(data,columns=['class_name','accuracy'])

    df_final=pd.concat([df_accuracy,df_joint2],axis=1)
    return df_final


def SaveJointValueSkeletons(sortedAverageConfPerson, acc_dic, class_name_list):
    #create dataframe from a dictionary of lists
    colSkeletonFirst=['S_0', 'S_1', 'S_2', 'S_3', 'S_4', 'S_5', 'S_6', 'S_7', 'S_8', 'S_9', 'S_10', 'S_11', 'S_12', 'S_13', 'S_14', 'S_15', 'S_16', 'S_17']      
    df_joint_first=pd.DataFrame.from_dict(sortedAverageConfPerson, orient='index', columns=colSkeletonFirst)
    # df_joint_second=pd.DataFrame.from_dict(sortedAverageConfSecondPerson, orient='index',columns=colSkeletonSecond)
    #add two dataframes horizontally
    df_joint2=df_joint_first.reset_index(drop=True) #remove the index and drop it so this and others can be combined


    _=list(acc_dic.items())
    data=zip(class_name_list, list(acc_dic.values()))
    df_accuracy=pd.DataFrame(data,columns=['class_name','accuracy'])

    df_final=pd.concat([df_accuracy,df_joint2],axis=1)
    return df_final


def mapConfidenceAccuracy(averageJointCOnf,acc_dic):
    #map the confidence value to the accuracy value
    new_dic={key:averageJointCOnf[key] for key in acc_dic.keys()}
    return new_dic


def calculateJointConfidence(numpyData,pickleData):
    #get average confidence value- joint based ---------------------------------------------------------------
    distri_dict={str(i):0 for i in range(number_class)}
    final_val={str(i):[0 for _ in  range(18)] for i in range(number_class)}
    for i in range(numpyData.shape[0]):
        label=pickleData[1][i] #index  0-399    
        data=numpyData[i,2,:,:,:]
        distri_dict[str(label)]+=1
        for i in range(18):
            final_val[str(label)][i]+=data[:,i,:].sum()/(300*2)
    return {str(i):[final_val[str(i)][j]/distri_dict[str(i)] for j in range(18)] for i in range(number_class)} # <-- average_conf_dict_per_joint=

def calculateJointConfidencePerPerson(numpyData,pickleData):
    #get average confidence value- joint based ---------------------------------------------------------------
    distri_dict={str(i):0 for i in range(number_class)}
    final_val={str(i):{'0':[0 for _ in  range(18)],'1':[0 for _ in  range(18)]} for i in range(number_class)}
    for i in range(numpyData.shape[0]):
        label=pickleData[1][i] #index  0-399    
        data=numpyData[i,2,:,:,:]
        distri_dict[str(label)]+=1
        for j in range(18):
            final_val[str(label)]['0'][j]+=data[:,j,0].sum()/(300)
            final_val[str(label)]['1'][j]+=data[:,j,1].sum()/(300)
    # return {str(i):{'0':[final_val[str(i)]['0'][j]/distri_dict[str(i)] for j in range(18)],'1':[final_val[str(i)]['1'][j]/distri_dict[str(i)] for j in range(18)]} for i in range(number_class)} # <-- average_conf_dict_per_joint=
    return {str(i):[final_val[str(i)]['0'][j]/distri_dict[str(i)] for j in range(18)] for i in range(number_class)}, {str(i):[final_val[str(i)]['1'][j]/distri_dict[str(i)] for j in range(18)] for i in range(number_class)} # <-- average_conf_dict_per_joint=

def plotAccuracyGraph(accuracy_dictionary,x_lables):
    acc_dic=accuracy_dictionary
    class_name_list=x_lables

    #plot a bar plot using matplotlib
    plt.figure(figsize=(60,5))
    plt.bar(range(number_class),list(acc_dic.values()),align='center')
    if showMyClasses:
        myEightCLasses=['48','156','68','83','330','30','255','57']
        acc_dic_selected={key:val for key,val in acc_dic.items() if key in myEightCLasses}
        acc_dic_selected_labels=[list(acc_dic.keys()).index(key) for key in acc_dic.keys() if key in myEightCLasses]
        plt.bar(acc_dic_selected_labels,list(acc_dic_selected.values()),align='center',color='#ff0000')
    # plt.xticks(range(number_class),list(acc_dic.keys()),rotation=90,size=5)
    plt.xticks(range(number_class),class_name_list,rotation=90,size=7)

    plt.grid(True)
    plt.savefig(imagePath,bbox_inches='tight')
    plt.show()
    


def classNameIndexMap(accuracy_dictionary):
    acc_dic=accuracy_dictionary
    labelmapIndex=loadTextFile(classNamePath)
    #split string into sub strings based on new line character
    allClasses=labelmapIndex.split('\n')
    _=allClasses[399]

    #find the indexes of a chracter in a string
    spacePlacesAllLines_list=[[pos for pos, char in enumerate(i) if char == ' '] for i in allClasses]
    labelmapIndex_list=[(i[:spacePlacesAllLines_list[ind][-1]],int(i[spacePlacesAllLines_list[ind][-1]+1:])) for ind,i in enumerate(allClasses)]
    # tmp=acc_dic.keys()
    classNamesList=[]
    for key in acc_dic.keys():
        for name,val in labelmapIndex_list:
            if val==int(key):
                classNamesList.append(name+' '+key)
                break
    return classNamesList


def classClassificatioAcc(dataframe):
    df=dataframe
    correct_dic={str(i):0 for i in range(number_class)} # class name goes from '0' to '399'
    all_dic={str(i):0 for i in range(number_class)}
    main_dic={'correct':correct_dic,'all':all_dic}

    for i in range(df.shape[0]):
        label=df.iloc[i]['real_label']
        predicted_label=df.iloc[i]['predicted_label']
        main_dic['all'][str(int(label))]+=1
        if label==predicted_label:
            main_dic['correct'][str(int(label))]+=1

    class_accuracy={str(i):(main_dic['correct'][str(i)]/main_dic['all'][str(i)]) for i in range(number_class)}  
    return class_accuracy, main_dic    

def checkVideoNone(data):
    a=0
    for i in range(data.shape[0]):
        if data[i].sum()==0:
            a+=1
    return a

def loadPickle(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

def loadNumpyAsmemoryMap(filename):
    return np.load(filename, mmap_mode='r')

def loadNumpy(filename):
    return np.load(filename)

def visualizeSkeleton(skeletonList):
    # inward_ori_index = [(4, 3), (3, 2), (7, 6), (6, 5), (13, 12), (12,11),
    #                 (10, 9), (9, 8), (11, 5), (8, 2), (5, 1), (2, 1),
    #                 (0, 1), (15, 0), (14, 0), (17, 15), (16, 14)]


    # inward_ori_index_change = [(4, 3), (3, 2), (7, 6), (6, 5), (13, 12), (12,11),
    #                 (10, 9), (9, 8), (11, 5), (8, 2), (5, 1), (2, 1),
    #                 (0, 1), (15, 0), (14, 0), (17, 15), (16, 14)]

    inward_ori_index = [(4, 3), (3, 2), (7, 6), (6, 5), (13, 12), (12,11),
                    (10, 9), (9, 8), (11, 1), (8, 1), (5, 1), (2, 1),
                    (0, 1), (15, 0), (14, 0), (17, 15), (16, 14)]


    inward_ori_index_change = [(4, 3), (3, 2), (7, 6), (6, 5), (13, 12), (12,11),
                    (10, 9), (9, 8), (11, 1), (8, 1), (5, 1), (2, 1),
                    (0, 1), (15, 0), (14, 0), (17, 15), (16, 14)]
    assert inward_ori_index == inward_ori_index_change #check if both are same ; it needs to be same 

    #test if graphs are same in model's graph and visualize graphs
    inward_ori_index_graph = [(4, 3), (3, 2), (7, 6), (6, 5), (13, 12), (12,11),
                    (10, 9), (9, 8), (11, 5), (8, 2), (5, 1), (2, 1),
                    (0, 1), (15, 0), (14, 0), (17, 15), (16, 14)]
    # assert inward_ori_index_graph== inward_ori_index_change

    available_edges=[]
    for edge in inward_ori_index:
        a,b=edge
        # a-=1
        # b-=1
        if skeletonList[2][a]==0 or skeletonList[2][b]==0:
            inward_ori_index_change.remove(edge)
    
    for drawableEdge in inward_ori_index_change:
        a,b=[i for i in drawableEdge]
        x_points=[skeletonList[0][a],skeletonList[0][b]]
        y_points=[skeletonList[1][a],skeletonList[1][b]]
        plt.plot(x_points,y_points,c='g')
        
    
    colors = 'g' #np.random.rand(N)
    confidenceJoints=np.asarray(skeletonList[2])
    area = (5 * confidenceJoints)**2  # 0 to 15 point radii
    plt.scatter(skeletonList[0],skeletonList[1],s=area,c=colors)
    for i in range(18):
        labelJoint=str(i)
        if skeletonList[2][i] !=0: #if confidence is not 0 (i.e., no joint detected)
            plt.text(skeletonList[0][i],skeletonList[1][i],labelJoint,c='g')
            pass

    plt.draw()
    # plt.savefig(savePath)
    plt.pause(.01)
    plt.clf()


def skeletonSelection(data, structure='coco'): #pass one sample at a time rather than the whole dataset
    typeA=type(data)
    data_numpy=data #TFEgerTesnor to numpy array
    size= data_numpy.shape[0]
    coco=[True]*8+ [False]*1+ [True]*10+ [False]*6
    no_foot=[True]*19+[False]*6 # mid hip is not there

    def forEachSample(data_numpy,structure):# processing
        pass
        if structure == 'coco':
            data_numpy=data_numpy[:,:,coco,:]
        elif structure == 'no_foot':
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
    return tmpNumpyt

def loadTextFile(filename):
    with open(filename, 'r') as f:
        return f.read()

if __name__=='__main__':
    main()
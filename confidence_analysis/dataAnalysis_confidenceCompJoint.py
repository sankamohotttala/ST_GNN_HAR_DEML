#this is to be used for comparison of confidence value  kinetics-400 based on the joint. 
#2 methods 1- 2 people skeletons and 2- combined person
#saves to csv but no plotting
import pickle
import numpy as np
from matplotlib import pyplot as plt

import pandas as pd
import seaborn

pathNumpy=r'F:\Codes\joint attention\2022 - comparison paper\kineticsSTGCN_code\result_analysis\Data\400_small\val_data.npy'
pathPickle=r'F:\Codes\joint attention\2022 - comparison paper\kineticsSTGCN_code\result_analysis\Data\400_small\val_label.pkl'

csvPath=r'F:\Codes\joint attention\2022 - comparison paper\kineticsSTGCN_code\result_analysis\CSV\data_30_kinetics400_small.csv'  #small set ~20,000
# csvPath=r'F:\Codes\joint attention\2022 - comparison paper\kineticsSTGCN_code\result_analysis\CSV\data_20_kinetics400_all.csv' #large set ~240,000

# saveCsvPath=r'F:\Codes\joint attention\2022 - comparison paper\kineticsSTGCN_code\result_analysis\Results_save\results_acc.csv'
saveCsvPath=r'F:\Codes\joint attention\2022 - comparison paper\kineticsSTGCN_code\result_analysis\Results_save\tmp_twoSkeleton.csv'
imagePath=r'F:\Codes\joint attention\2022 - comparison paper\kineticsSTGCN_code\result_analysis\Results_save\tmp1.png'

classNamePath=r'F:\Codes\joint attention\2022 - comparison paper\kineticsSTGCN_code\result_analysis\labelnamesIndex.txt'



number_class=400 
sort =True #sort the accuracy values of classes
showMyClasses=True #shows the 8 classes in red

def main():

    #new implementation
    pickleData=loadPickle(pathPickle) #19796 data points
    numpyData=loadNumpy(pathNumpy)
    numpyData2=loadNumpyAsmemoryMap(pathNumpy)
    noneNum=checkVideoNone(numpyData)  # 109 videos with no joint detected
    skeletonVisual=toVisualizeSkeleton(structure='coco') #for skeleton structure visualization 

    averageJointCOnf=calculateJointConfidence(numpyData,pickleData) #for combined persons (average of 2 persons)
    averageJointFirstPersonConf,averageJointSecondPersonConf=calculateJointConfidencePerPerson(numpyData,pickleData) #considers each person seperately

    df =pd.read_csv(csvPath)
    acc_dic,main_dic=classClassificatioAcc(df) 
    #sort a dictionary by value
    if sort:
        acc_dic=sorted(acc_dic.items(), key=lambda x: x[1], reverse=True)
        #convert a list of tuples to a dictionary
        acc_dic=dict(acc_dic)
    class_name_list=classNameIndexMap(acc_dic) #for the use of display in plot ; if sort-True , then this is the list for the sorted one

    # plotAccuracyGraph(acc_dic,class_name_list)
    sortedAverageConf=mapConfidenceAccuracy(averageJointCOnf,acc_dic)
    sortedAverageConfFirstPerson=mapConfidenceAccuracy(averageJointFirstPersonConf,acc_dic)
    sortedAverageConfSecondPerson=mapConfidenceAccuracy(averageJointSecondPersonConf,acc_dic)

    #visualize the skeleton with joint confidence
    averageConfJointClass=sortedAverageConf['330'] #class need to visualize
    visualizeSkeletonJointsConfidence(skeletonVisual,averageConfJointClass)

    df_final_save=SaveJointValuesTwoSkeletons(sortedAverageConfFirstPerson, sortedAverageConfSecondPerson, acc_dic, class_name_list)
    # df_final_save=SaveJointValueSkeletons(sortedAverageConf, acc_dic, class_name_list)
    df_final_save.to_csv(saveCsvPath,index=False)



def toVisualizeSkeleton(structure): #for the use of result visualization
    dataPath=r'F:\Codes\joint attention\2022 - comparison paper\kineticsSTGCN_code\data\st-gcn-processed-data\data\Kinetics\5_class_25_balancedChild\train_data.npy'
    if structure=='coco':
        coco=[True]*8+ [False]*1+ [True]*10+ [False]*6
        # dataPath=r'F:\Codes\joint attention\2022 - comparison paper\kineticsSTGCN_code\data\st-gcn-processed-data\data\Kinetics\kinetics-skeleton\fourClassesTest\child\val_data.npy'
        data=np.load(dataPath)
        # skeleleton= data[2,:,7,coco,0]  #contains all the joints of the 8th frame
        skele_data=data[8,:,6,:,0] 
        skeleleton= skele_data[:,coco] 
    elif structure=='body_25':
        # dataPath=r'F:\Codes\joint attention\2022 - comparison paper\kineticsSTGCN_code\data\st-gcn-processed-data\data\Kinetics\5_class_25_balancedChild\train_data.npy'
        data=np.load(dataPath)
        skeleleton= data[8,:,6,:,0] #contains all the joints of the 8th frame 
    skeleleton[1,:]=skeleleton[1,:]*(-1) #flip the skeleton verticly

    return skeleleton
    

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

def visualizeSkeleton(skeletonList): #for generic visualization of skeletons
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


def visualizeSkeletonJointsConfidence(skeletonList,averageConfidence): #for average conf visualization of skeleton joints
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
        
    
    colors = 'b' #np.random.rand(N)
    confidenceJoints=np.asarray(averageConfidence)
    area = (10 * confidenceJoints)**2  # 0 to 15 point radii
    plt.scatter(skeletonList[0],skeletonList[1],s=area,c=colors)
    for i in range(18):
        labelJoint=str(i)
        if skeletonList[2][i] !=0: #if confidence is not 0 (i.e., no joint detected)
            plt.text(skeletonList[0][i],skeletonList[1][i],labelJoint,c='g')
        else:
            plt.text(skeletonList[0][i],skeletonList[1][i],labelJoint,c='r')
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
#display box and wisker in different ways for same csv file
#display for correctly classifed case and misclassfy case

import pandas as pd
from matplotlib import pyplot as plt
import seaborn

pathCSV='all_4_methods/8/all_4.csv'

status=0

def main():
    df =pd.read_csv(pathCSV)
    if status==0: #all data
        final_df=df
    elif status==1: #only correctly classified
        final_df=df.loc[df['real_label']==df['predicted_label']]
        print(final_df.head)
    elif status==2: #only incorrectly classified
        final_df=df.loc[df['real_label']!=df['predicted_label']]
        print(final_df.head)
    elif status==3: # both misclassify and classify in same plot
        newList=newListClassify(df)
        df['Result']=newList
        final_df=df

    seaborn.set(style='whitegrid')

    # if status!=3:
    #     seaborn.boxplot(x='real_label',y='real_probability',data=final_df,palette='Set3')#,dodge=True
    #     # seaborn.despine(offset=10, trim=True)
    # else:
    #     seaborn.boxplot(x='real_label',y='real_probability',hue='Result',data=final_df,palette='Set3',dodge=True)#,dodge=True
    #     # seaborn.despine(offset=10, trim=True)


    seaborn.boxplot(x='real_label',y='real_probability',hue='Method',data=final_df,palette='Set3',dodge=True)#,dodge=True
    # seaborn.despine(offset=10, trim=True)


    
    plt.show()

    pass

def newListClassify(dataframe):
    listCol=[]
    for index, row in dataframe.iterrows():
        if row['real_label'] == row['predicted_label']:
            listCol.append('Correct')
        else:
            listCol.append('Type 1')
    
    return listCol
            

if __name__=='__main__':
    main()

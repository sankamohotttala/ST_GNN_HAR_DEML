from re import I
from matplotlib import pyplot as plt
import statistics
from math import sqrt
import pandas as pd

pathCSV='3_class/adult_data_30.csv'
numClasses=3

def plot_confidence_interval(x, values, z=1.96, color='#2187bb', horizontal_line_width=0.25):
    mean = statistics.mean(values)
    stdev = statistics.stdev(values)
    confidence_interval = z * stdev / sqrt(len(values))

    left = x - horizontal_line_width / 2
    top = mean - confidence_interval
    right = x + horizontal_line_width / 2
    bottom = mean + confidence_interval
    plt.plot([x, x], [top, bottom], color=color)
    plt.plot([left, right], [top, top], color=color)
    plt.plot([left, right], [bottom, bottom], color=color)
    plt.plot(x, mean, 'o', color='#f44336')

    return mean, confidence_interval

labelPlaceList=[i+1 for i in range(numClasses)]
labelValList=[str(i)+'_class' for i in range(numClasses)]

df =pd.read_csv(pathCSV)
allClassesList=[]
for i in range(numClasses):
  tmp=df.loc[df['real_label'] == i, 'real_probability']
  allClassesList.append(tmp)


plt.xticks(labelPlaceList,labelValList)
plt.title('Confidence Interval')
for i in range(numClasses):
  plot_confidence_interval(i+1, allClassesList[i])  

plt.grid(True)
plt.show()
pass
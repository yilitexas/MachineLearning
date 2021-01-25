import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix

#read csv file
flow1 = pd.read_csv('cleaned_training1.csv')


#get the Series for each of the columns
fsca = flow1["FSC-A"]
ssca = flow1["SSC-A"]
fsch = flow1["FSC-H"]
ssch = flow1["SSC-H"]
fscw = flow1["FSC-W"]
sscw = flow1["SSC-W"]
gfp = flow1["GFP-A"]
mkate = flow1["mCherry-A"]

state = flow1["STATE"]

#build the DataFrame for the features: X
input4 = pd.DataFrame({'FSC-A': fsca, 'SSC-A': ssca, 'FSC-H': fsch, 'SSC-H': ssch, 'FSC-W': fscw, 'SSC-W': sscw})

#build the DataFrame for the labels: Y
output4 = pd.DataFrame({'STATE': state})

#build K-means model and do clustering
kmeans3 = KMeans(n_clusters=2)
kmeans3.fit(input4)

#find the cluster center's coordinates
print(kmeans3.cluster_centers_)

#print the cluster number for each sample
label2 = kmeans3.labels_
print(label2)

output15 = pd.DataFrame({'FSC-A': fsca, 'SSC-A': ssca, 'FSC-H': fsch, 'SSC-H': ssch, 'FSC-W': fscw,
'SSC-W': sscw, 'GFP-A': gfp, 'mCherry-A': mkate, 'STATE': state, 'cluster': label2})
output15.to_csv(r'two_clusters_results.csv', index = None, header=True)

color2 = []
len2 = len(state)

gfp2 = []
mkate2 = []

for i in range(len2):
    if ((mkate[i] > 0) and (gfp[i] > 0)):
        temp1 = np.log10(gfp[i])
        temp2 = np.log10(mkate[i])

        if (label2[i] == 1):
            temp3 = 'red'

        elif (label2[i] == 0):
            temp3 = 'green'

        else:
            temp3 = 'blue'

        gfp2.append(temp1)
        mkate2.append(temp2)
        color2.append(temp3)

plt.scatter(gfp2, mkate2, c=color2, cmap='rainbow', s=0.1)

plt.show()

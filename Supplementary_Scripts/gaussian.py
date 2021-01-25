import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.mixture import GaussianMixture

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

gmm2 = GaussianMixture(n_components=2)
gmm2.fit(input4)


#find the cluster center's coordinates
#print(gmm2.cluster_centers_)

#print the cluster number for each sample
label1 = gmm2.predict(input4)
print(label1)

color1=[]
len1=len(state)

gfp1=[]
mkate1=[]


for i in range(len1):
    if ((mkate[i]>0) and (gfp[i]>0)):
        temp1 = np.log10(gfp[i])
        temp2 = np.log10(mkate[i])

        if (label1[i] == 2):
            temp3 = 'green'

        elif (label1[i] == 1):
            temp3 = 'yellow'

        elif (label1[i] == 0):
            temp3 = 'red'

        else:
            temp3 = 'blue'


        gfp1.append(temp1)
        mkate1.append(temp2)
        color1.append(temp3)


plt.scatter(gfp1, mkate1, c=color1, cmap='rainbow', s=0.1)
plt.show()


output14 = pd.DataFrame({'FSC-A': fsca, 'SSC-A': ssca, 'FSC-H': fsch, 'SSC-H': ssch, 'FSC-W': fscw,
'SSC-W': sscw, 'GFP-A': gfp, 'mCherry-A': mkate, 'STATE': state, 'cluster': label1})
output14.to_csv(r'two_clusters_results.csv', index = None, header=True)





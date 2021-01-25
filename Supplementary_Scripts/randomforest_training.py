import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
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

state = flow1["STATE"]

#build the DataFrame for the features: X
input4 = pd.DataFrame({'FSC-A': fsca, 'SSC-A': ssca, 'FSC-H': fsch, 'SSC-H': ssch, 'FSC-W': fscw, 'SSC-W': sscw})

#build the DataFrame for the labels: Y
output4 = pd.DataFrame({'STATE': state})



for i in range(100):
    temp1 = i + 1
    print(str(temp1))
    name1 = 'randomforest_cv10_' + str(temp1) + '.pkl'
    
    #Create a Gaussian Classifier
    clf = RandomForestClassifier(n_estimators=temp1)

    scores4 = cross_val_score(clf, input4, output4.values.ravel(), cv=10)
    #print(type(scores4))
    mean1 = np.mean(scores4)

    if (mean1 > 0.8):
        print(scores4)
        print(str(mean1))
        print(str(np.std(scores4)))

        # Save the model as a pickle in a file
        clf.fit(input4,output4)
        joblib.dump(clf, name1)


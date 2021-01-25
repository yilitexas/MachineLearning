import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import mean_squared_error, r2_score
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

logReg = LogisticRegression()

scores4 = cross_val_score(logReg, input4, output4.values.ravel(), cv=10)
mean1 = np.mean(scores4)

print(scores4)
print(str(mean1))
print(str(np.std(scores4)))

logReg.fit(input4,output4)

#read csv file
flow2= pd.read_csv('testing_mix1.csv')


#get the Series for each of the columns
fsca2 = flow2["FSC-A"]
ssca2 = flow2["SSC-A"]
fsch2 = flow2["FSC-H"]
ssch2 = flow2["SSC-H"]
fscw2 = flow2["FSC-W"]
sscw2 = flow2["SSC-W"]

state2 = flow2["STATE"]

#build the DataFrame for the features: X
input5 = pd.DataFrame({'FSC-A': fsca2, 'SSC-A': ssca2, 'FSC-H': fsch2, 'SSC-H': ssch2, 'FSC-W': fscw2, 'SSC-W': sscw2})

#build the DataFrame for the labels: Y
output5 = pd.DataFrame({'STATE': state2})

###predict on the test dataset
y_predict5 = logReg.predict(input5)

# Model Accuracy, how often is the classifier correct?
accuracy = accuracy_score(output5, y_predict5)

outmatrix1 = confusion_matrix(output5, y_predict5)

print(outmatrix1)


import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.externals import joblib
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import VotingClassifier

#read csv file
flow1 = pd.read_csv('testing_mix1.csv')


#get the Series for each of the columns
fsca = flow1["FSC-A"]
ssca = flow1["SSC-A"]
fsch = flow1["FSC-H"]
ssch = flow1["SSC-H"]
fscw = flow1["FSC-W"]
sscw = flow1["SSC-W"]

state = flow1["STATE"]

#build the DataFrame for the features: X
input3 = pd.DataFrame({'FSC-A': fsca, 'SSC-A': ssca, 'FSC-H': fsch, 'SSC-H': ssch, 'FSC-W': fscw, 'SSC-W': sscw})

#build the DataFrame for the labels: Y
output3 = pd.DataFrame({'STATE': state})

###################################################################
#read csv file
flow2 = pd.read_csv('cleaned_training1.csv')


#get the Series for each of the columns
fsca2 = flow2["FSC-A"]
ssca2 = flow2["SSC-A"]
fsch2 = flow2["FSC-H"]
ssch2 = flow2["SSC-H"]
fscw2 = flow2["FSC-W"]
sscw2 = flow2["SSC-W"]

state2 = flow2["STATE"]


#build the DataFrame for the features: X
input4 = pd.DataFrame({'FSC-A': fsca2, 'SSC-A': ssca2, 'FSC-H': fsch2, 'SSC-H': ssch2, 'FSC-W': fscw2, 'SSC-W': sscw2})

#build the DataFrame for the labels: Y
output4 = pd.DataFrame({'STATE': state2})

cnn_20_14 = MLPClassifier(solver='adam', alpha=1e-3, hidden_layer_sizes=(20, 14), random_state=1, max_iter=1000)
cnn_23_13 = MLPClassifier(solver='adam', alpha=1e-3, hidden_layer_sizes=(23, 13), random_state=1, max_iter=1000)
cnn_26_9 = MLPClassifier(solver='adam', alpha=1e-3, hidden_layer_sizes=(26, 9), random_state=1, max_iter=1000)
cnn_29_4 = MLPClassifier(solver='adam', alpha=1e-3, hidden_layer_sizes=(29, 4), random_state=1, max_iter=1000)
knn_6 = KNeighborsClassifier(n_neighbors=6)
knn_8 = KNeighborsClassifier(n_neighbors=8)
rf_6 = RandomForestClassifier(n_estimators=6)
rf_8 = RandomForestClassifier(n_estimators=8)
rf_12 = RandomForestClassifier(n_estimators=12)
rf_16 = RandomForestClassifier(n_estimators=16)

estimators1=[('rf1', rf_6), ('rf2', rf_8), ('rf3', rf_12), ('rf4', rf_16), ('knn1', knn_6), ('knn2', knn_8), ('mlp1', cnn_20_14), ('mlp2', cnn_23_13), ('mlp3', cnn_26_9), ('mlp4', cnn_29_4)]

eclf1 = VotingClassifier(estimators=estimators1, voting='hard')
eclf1 = eclf1.fit(input4, output4)

joblib.dump(eclf1, "hard_vote.pkl")


#predict on the test dataset
y_predict3 = eclf1.predict(input3)

# Model Accuracy, how often is the classifier correct?
accuracy = accuracy_score(output3, y_predict3)

outmatrix1 = confusion_matrix(output3, y_predict3)

print(outmatrix1)

precision = outmatrix1[0][0]/(outmatrix1[0][0] + outmatrix1[1][0])
recall = outmatrix1[0][0]/(outmatrix1[0][0] + outmatrix1[0][1])
fvalue = (2 * precision * recall)/(precision + recall)

print(str(precision))
print(str(recall))
print(str(fvalue))


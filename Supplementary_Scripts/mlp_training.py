import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.externals import joblib
from sklearn.neural_network import MLPClassifier
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

count = 0

for num1 in range(30):
    for num2 in range(30):
        count = count + 1
        print(str(count))
        temp1 = num1 + 1
        temp2 = num2 + 1
        name1 = 'mlp_cv10_' + str(temp1) + '_' + str(temp2) + '.pkl'

        #build a MLP model and train the model using test dataset
        clf = MLPClassifier(solver='adam', alpha=1e-3, hidden_layer_sizes=(temp1, temp2), random_state=1, max_iter=1000)

        scores4 = cross_val_score(clf, input4, output4.values.ravel(), cv=10)
        #print(type(scores4))
        mean1 = np.mean(scores4)
        if (mean1 > 0.8):
            print(str(temp1))
            print(str(temp2))

            print(scores4)
            print(str(mean1))
            print(str(np.std(scores4)))

            clf.fit(input4,output4)

            joblib.dump(clf, name1)



import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import svm
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

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

model1 = SVC()
# defining parameter range
param_grid1 = {'C': [0.001, 0.01, 0.1, 1, 10], 'kernel': ['linear']}

grid1 = GridSearchCV(model1, param_grid1, refit=True, verbose=3)

# fitting the model for grid search
grid1.fit(input4, output4)

# print best parameter after tuning
print(grid1.best_params_)

# print how our model looks after hyper-parameter tuning
print(grid1.best_estimator_)

joblib.dump(grid1, "svm_linear.pkl")


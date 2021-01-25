import pandas as pd
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

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
gfp = flow1["GFP-A"]
mkate = flow1["mCherry-A"]

features = ['FSC-A', 'FSC-H', 'FSC-W', 'SSC-A', 'SSC-H', 'SSC-W']
input4 = flow1.loc[:, features].values
output4 = flow1.loc[:,['STATE']].values
input4 = StandardScaler().fit_transform(input4)


input4_embedded = TSNE(n_components=2).fit_transform(input4)
label1 = input4_embedded[:, 0]
label2 = input4_embedded[:, 1]

output14 = pd.DataFrame({'FSC-A': fsca, 'SSC-A': ssca, 'FSC-H': fsch, 'SSC-H': ssch, 'FSC-W': fscw, 'SSC-W': sscw,
'GFP': gfp, 'MKATE': mkate, 'STATE': state, 'LABEL1': label1, 'LABEL2': label2})
output14.to_csv(r'tnse_results2.csv', index = None, header=True)
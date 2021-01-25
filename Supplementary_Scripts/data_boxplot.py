import matplotlib.pyplot as plt
import pandas as pd

#read csv file
flow1 = pd.read_csv('cleaned_training1.csv')


#get the Series for each of the columns
fsca = flow1["FSC-A"]
ssca = flow1["SSC-A"]
fsch = flow1["FSC-H"]
ssch = flow1["SSC-H"]
fscw = flow1["FSC-W"]
sscw = flow1["SSC-W"]

fig1, ax1 = plt.subplots()
ax1.set_title('Feature Distributions')
data = [fsca, ssca, fsch, ssch, fscw, sscw]

ax1.boxplot(data)
ax1.set_xticklabels(['FSC-A', 'SSC-A', 'FSC-H', 'SSC-H', 'FSC-W', 'SSC-W'])

plt.show()

print(flow1.mean())
print(flow1.std())

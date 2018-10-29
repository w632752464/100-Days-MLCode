
import pandas as pd

dataset = pd.read_csv('Data.csv')
# X = dataset.iloc[:, :].values    #对行处理
# Y = dataset.iloc[:, 3].values      #队列处理

X = dataset.loc[:].values    #对行处理
Y = dataset.loc[:].values      #队列处理
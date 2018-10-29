
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

dataset = pd.read_csv('studentScores.csv')
X = dataset.iloc[:, :1].values  # 对行处理
Y = dataset.iloc[:, 1].values   # 对列处理

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=0)

'使训练集拟合简单线性回归模型'
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor = regressor.fit(X_train, Y_train)

'预测结果'
# 使用训练好的模型regressor对X_test进行预测
Y_pred = regressor.predict(X_test)
ar = np.array(Y_pred).reshape(7, 1)
rs = np.hstack((X_test, ar))
print(rs)


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

'数据预处理'
dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:, [2, 3]].values
Y = dataset.iloc[:, 4].values

'训练数据'
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=0)

'特征标准化'
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

'Logistic 模型'
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier = classifier.fit(X_train, Y_train)

'预测结果'
Y_pred = classifier.predict(X_test)
ar = np.array(Y_pred).reshape(100, 1)
rs = np.hstack((X_test, ar))
print(rs)

'构造混合矩阵'
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, Y_pred)
print(cm)

from matplotlib.colors import ListedColormap
X_set, Y_set = X_train, Y_train
# meshgrid函数将两个输入的数组进行扩展
# arange函数用于创建等差数组,返回一个array对象
# contourf函数用于填充图
X1, X2 = np. meshgrid(np.arange(start=X_set[:, 0].min()-1, stop=X_set[:, 0].max()+1, step=0.01),
                      np.arange(start=X_set[:, 1].min()-1, stop=X_set[:, 1].max()+1, step=0.01))
# ravel函数是将矩阵变为一个一维的数组，其中X1.ravel()就表示x轴的坐标，X2.ravel()就表示了y轴的坐标，
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha=0.75, cmap=ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())    # 设置图的边界
plt.ylim(X2.min(), X2.max())
# enumerate() 函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据和数据下标
for i, j in enumerate(np.unique(Y_set)):    # np.unique()找到数组的唯一元素。
    plt.scatter(X_set[Y_set == j, 0], X_set[Y_set == j, 1],
                c=ListedColormap(('red', 'green'))(i), label=j)
plt. title('LOGISTIC(Training set)')
plt. xlabel('Age')
plt. ylabel('Estimated Salary')
plt. legend()
plt. show()

X_set, Y_set = X_test, Y_test
X1, X2 = np. meshgrid(np.arange(start=X_set[:, 0].min()-1, stop=X_set[:, 0].max()+1, step=0.01),
                      np.arange(start=X_set[:, 1].min()-1, stop=X_set[:, 1].max()+1, step=0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha=0.75, cmap=ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np. unique(Y_set)):
    plt.scatter(X_set[Y_set == j, 0], X_set[Y_set == j, 1],
                c=ListedColormap(('red', 'green'))(i), label=j)
plt. title('LOGISTIC(Test set)')
plt. xlabel('Age')
plt. ylabel('Estimated Salary')
plt. legend()
plt. show()

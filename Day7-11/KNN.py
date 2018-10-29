import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

'数据预处理'
dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:, [2, 3]].values
Y = dataset.iloc[:, 4].values

'训练数据'
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=0)

'提取特征'
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

'使用KNN拟合数据'
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier() # default: n_neighbors=5, metric='minkowski'（用于树的距离度量）, p=2(代表使用欧氏距离）
classifier.fit(X_train, Y_train)

'预测测试集的结果'
Y_pred = classifier.predict(X_test) # 为所提供的数据（X_test）预测类标签

'构造混合矩阵'
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, Y_pred)   # 计算混淆矩阵来评估分类的准确性
print(cm)

'训练集可视化'
from matplotlib.colors import ListedColormap
X_set1, Y_set1 = X_train, Y_train
'''==========================================================
# meshgrid函数详细用法见meshgrid.py
# arange函数用于创建等差数组,返回一个array对象
=========================================================='''
X1, X2 = np.meshgrid(np.arange(start=X_set1[:, 0].min() - 1, stop=X_set1[:, 0].max() + 1, step=0.01),
                     np.arange(start=X_set1[:, 1].min() - 1, stop=X_set1[:, 1].max() + 1, step=0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha=0.75, cmap=ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(Y_set1)):
    plt.scatter(X_set1[Y_set1 == j, 0], X_set1[Y_set1 == j, 1],
                c=ListedColormap(('red', 'green'))(i), label=j)
plt.title('SVM(Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

'测试集可视化'
X_set2, Y_set2 = X_test, Y_test
X1, X2 = np.meshgrid(np.arange(start=X_set2[:, 0].min() - 1, stop=X_set2[:, 0].max() + 1, step=0.01),
                     np.arange(start=X_set2[:, 1].min() - 1, stop=X_set2[:, 1].max() + 1, step=0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha=0.75, cmap=ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(Y_set2)):
    plt.scatter(X_set2[Y_set2 == j, 0], X_set2[Y_set2 == j, 1],
                c=ListedColormap(('red', 'green'))(i), label=j)
plt.title('SVM(Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

# 先导入pandas,csv,sklearn,scipy包
import pandas as pd
import matplotlib.pyplot as plt

'导入数据'
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values    #对行处理
Y = dataset.iloc[:, 3].values      #队列处理

'''--------------------------------------------------------------------------------------------------------------------
pandas 中关于loc跟iloc总结：
对行处理：pandas.loc[1:5]是从1到5行数据 ， pandas.iloc[1:5]是从1到4行数据。
对列处理：pandas.loc对列是不可以切割的。他只能根据列名来取列数据；如：pandas.loc[1:5，’列名’]
         pandas.iloc可以不用指定列名，如pandas.iloc[1:5,1] 是指取出第一列数据。
         pandas.iloc对列进行位置切割，开始位置为第一列之前，指定其为0，如：pandas.iloc[1:5,0:1]是指从位置为0跟1的地方切片，
         取出数据，即为第一列数据，pandas.iloc[1:5,1:2] 是指从位置为1跟2的地方切割，即为第二列数据。
---------------------------------------------------------------------------------------------------------------------'''

'处理丢失数据(实现对一个数据中缺失数据的填补，在缺失处放上平均值。)'
# sklearn用于对文本特征提取/预处理数据
# fit方法的主要工作是获取特征信息和目标值信息。
# transform方法主要用来对特征进行转换。
from sklearn.preprocessing import Imputer
imputer = Imputer()   # imputer默认值情况为：imputer（missing_values= 'NaN', strategy= 'mean', axis = 0）
imputer = imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])   # 填补第X中所有缺失值
print(X)

'编码分类数据'
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()   # 调用这个编辑标签的方法
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])   # 做标签记录

'''=============================================================================================
# LabelEncoder:对分类型特征值进行编码，即对不连续的数字或者文本进行编号。常用功能：
# fit(y) ：fit可看做一本空字典，y可看作要塞到字典中的词。
# transform(y) ：将y转变成索引值。
# fit_transform(y)：相当于先进行fit再进行transform，即把y塞到字典中去以后再进行transform得到索引值。
================================================================================================'''

'创建自变量'
# OneHotEncoder用于将表示分类的数据扩维
onehotencoser = OneHotEncoder(categorical_features=[0]) # 指定第一列的特征进行分类
X = onehotencoser.fit_transform(X).toarray()

'创建因变量'
labelencoder_Y = LabelEncoder()
Y = labelencoder_Y.fit_transform(Y)

'数据集分割为训练集和测试集'
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
# random_state 设置相同，那么当别人重新运行你的代码的时候就能得到完全一样的结果，复现和你一样的过程。

'特征缩放'
# 标准化数据，保证每个维度的特征数据方差为1，均值为0。使得预测结果不会被某些维度过大的特征值而主导
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

'''(官方文档)fit方法是用于从一个训练集中学习模型参数，其中就包括了归一化时用到的均值，标准偏差。
transform方法就是用于将模型用于位置数据，fit_transform就很高效的将模型训练和转化合并到一起，训练样本先做
fit，得到mean，standard deviation，然后将这些参数用于transform（归一化训练数据），使得到的训练数据是归一化的，
而测试数据只需要在原先得到的mean，std上来做归一化就行了，所以用transform就行了。'''

print(X_train)
print(X_test)

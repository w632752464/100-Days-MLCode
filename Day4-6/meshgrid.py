import numpy as np

nx, ny = (3, 2)
# 从0开始到1结束，返回一个numpy数组,nx代表数组中元素的个数
x = np.linspace(0, 1, nx)
# [ 0.   0.5  1. ]
y = np.linspace(0, 1, ny)
# [0.  1.]
xv, yv = np.meshgrid(x, y)
'''
xv
[[ 0.   0.5  1. ]
 [ 0.   0.5  1. ]]
 yv
 [[ 0.  0.  0.]
  [ 1.  1.  1.]]
'''
'''meshgrid函数将两个输入的数组x和y进行扩展，前一个的扩展与后一个有关，后一个的扩展与前一个有关，
    前一个是竖向扩展，后一个是横向扩展。因为，y的大小为2，所以x竖向扩展为原来的两倍，而x的大小为3，
    所以y横向扩展为原来的3倍。通过meshgrid函数之后，输入由原来的数组变成了一个矩阵。'''
nx, ny = (3, 3)
# 从0开始到1结束，返回一个numpy数组,nx代表数组中元素的个数
x = np.linspace(0, 1, nx)
print(x)
# [ 0.   0.5  1. ]
y = np.linspace(1, 2, ny)
print(y)
# [ 1.   1.5  2. ]
xv, yv = np.meshgrid(x, y, sparse=True)
print(xv)
# [[ 0.  0.5  1.]]
print(yv)
'''
[[ 1.]
 [ 1.5]
 [ 2.]]
'''


import numpy as np
# nx, ny = (3, 3)
# # 从0开始到1结束，返回一个numpy数组,nx代表数组中元素的个数
# x = np.linspace(0, 2, nx)
# print(x)
# # [ 0. 1.  2. ]
# y = np.linspace(0, 3, ny)
# print(y)
# # [ 0.   1.5  3. ]
# xv, yv = np.meshgrid(x, y)
# print(xv.ravel())
# # [ 0.  1.  2.  0.  1.  2.  0.  1.  2.]
# print(yv.ravel())
# # [ 0.  0.  0.  1.5  1.5  1.5  3.  3.  3.]

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


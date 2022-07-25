# #可视化卷积
# import cv2
# import tensorflow as tf
# import matplotlib.pyplot as plt
# def func19(img_path):
#
#     #读取图片，矩阵化，转换为张量
#     img_data=cv2.imread(img_path)
#     tf.compat.v1.disable_v2_behavior()
#     tf.compat.v1.get_default_graph()
#     img_data=tf.constant(img_data,dtype=tf.float32)
#     print(img_data.shape)
#     #将张量转化为4维
#     img_data=tf.reshape(img_data,shape=[1,454,700,3])
#     print(img_data.shape)
#     #权重
#     weights= tf.Variable(tf.random.uniform(shape=[2, 2, 3, 3], dtype=tf.float32))
#     #卷积
#     conv=tf.nn.conv2d(img_data,weights,strides=[1,3,3,1],padding='SAME')
#     print(conv.shape)
#     img_conv=tf.reshape(conv,shape=[152,234,3])
#     print(img_conv.shape)
#     img_conv=tf.nn.relu(img_conv)
#     with tf.compat.v1.Session() as sess:
#
#         #全局初始化
#         sess.run(tf.compat.v1.global_variables_initializer())
#         img_conv=sess.run(img_conv)
#         plt.title('conv')
#         plt.imshow(img_conv,cmap='gray')
#         plt.show()
#
#
# if __name__== '__main__':
#     img_path='E:/python/tools/common/P02-0148.png'
#     func19(img_path)
#
#
#

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

#可视化图像像素矩阵
img=np.array(Image.open('E:/python/tools/common/P02-0148.png'))
n=img.shape[0]
m=img.shape[1]
plt.figure(figsize=(n, m))
for krow, row in enumerate(img):
    for kcol, num in enumerate(row):
        plt.text(10*kcol+5, 10*krow+5, num,horizontalalignment='center',verticalalignment='center')
plt.grid(linestyle="solid")
plt.axis([0, 10*n, 10*m, 0])

plt.xticks(np.linspace(0, 10*n, n + 1), [])

plt.yticks(np.linspace(0, 10*m, m + 1), [])
plt.show()

# import numpy as np
#
# from matplotlib import rcParams, pyplot as plt
#
# rcParams['font.family'] = 'serif'
#
# rcParams['font.size'] = 16
#
# A = [[1, 3, 4, 5, 6, 7],
#
# [3, 3, 0, 7, 9, 2],
#
# [1, 3, 4, 5, 6, 6]]
#
# X = ["A", "B", "C", "E", "F", "G"]
#
# Y = ["R", "S", "T"]
#
# m = len(Y)
#
# n = len(X)
#
# plt.figure(figsize=(n + 1, m + 1))
#
# for krow, row in enumerate(A):
#
#     plt.text(5, 10*krow + 15, Y[krow],horizontalalignment='center',verticalalignment='center')
#
#     for kcol, num in enumerate(row):
#
#         if krow == 0:
#
#             plt.text(10*kcol + 15, 5, X[kcol], horizontalalignment='center',verticalalignment='center')
#
#         plt.text(10*kcol + 15, 10*krow + 15, num,horizontalalignment='center',verticalalignment='center')
#
# plt.axis([0, 10*(n + 1), 10*(m + 1), 0])
#
# plt.xticks(np.linspace(0, 10*(n + 1), n + 2), [])
#
# plt.yticks(np.linspace(0, 10*(m + 1), m + 2), [])
#
# plt.grid(linestyle="solid")
#
# plt.savefig("Table.png", dpi=300)
#
# plt.show()

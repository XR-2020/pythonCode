import os
import numpy as np
from matplotlib import pyplot as plt

"""
在一张图中同时画出多个网络的损失函数
"""
def data(file):
    # 读取文件每一行数据
    sourceInLine = file.readlines()
    flag=0
    conv=[]
    attention= []
    for line in sourceInLine:
        temp1 = line.strip('\n')
        temp2 = temp1.split()
        if (line == '\n'):
            flag = 1
            continue
        if flag==0:
            conv.append(temp2[1])
        else:
            attention.append(temp2[1])

    return conv,attention

if __name__ == '__main__':
    file=open('../../Learning/data.txt')
    conv,attention=data(file)
    conv=np.array(conv).astype(float)
    attention=np.array(attention).astype(float)
    epoch=[x for x in range(1,121)]
    for i in epoch:
        plt.scatter(i, conv[i-1], s=0.1, color='green')
        plt.scatter(i, attention[i-1], s=8, color='red')
    plt.plot(epoch, conv, linewidth=1, color='g',label='Ordinary Convolution')
    plt.plot(epoch, attention, linewidth=1, color='r',label='Attention Convolution')
    plt.legend()
    plt.title('loss', fontsize=20)
    plt.show()
import numpy as np
"""
    图像分辨率为iw＊ih， 卷积核大小为fw＊fh
"""

#------------0填充-------------#
def Convolve0(Img,iw,ih,fw,fh):
    #---------创建新全0数组-------#
    New_I = np.zeros((iw+fw-1,ih+fh-1)) .astype(np.float32)
    #--------原图像像素值对应到新图像中----#
    for i in range (iw):
        for j in range(ih):
            New_I[i,j+fh-1]=Img[i,j]
#------------镜像填充-------------#
def Convolve_mirror(Img,iw,ih,fw,fh):
    # ---------创建新全0数组-------#
    New_I = np.zeros((iw + fw - 1, ih + fh - 1)).astype(np.float32)
    # --------原图像像素值对应到新图像中----#
    for i in range(iw):
        for j in range(ih):
            New_I[i, j + fh - 1] = Img[i, j]
    #-------将新增的行做镜像对称--------#
    for i in range (1,fw):
         #New_I[-i,:]=New_I[-(2*fw-1)+i,:]
         New_I[-i, :] = New_I[-(2 * fw ) + i, :]
    # -------将新增的列做镜像对称--------#
    for j in range(0,fh-1):
         #New_I[:,j]=New_I[:,2*fh-3-j]
         New_I[:, j] = New_I[:, 2 * fh - 2 - j]
    #print('镜像对称后:\n',New_I)


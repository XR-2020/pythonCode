import os
import cv2 as cv
import numpy as np
mask=cv.imread('mask.png',0)
mask=(mask/255).astype(np.uint8)
cat=cv.imread('cat.png',0)
house=cv.imread('img.png',0)
noise=cv.imread('noise .png',0)
sum=cv.addWeighted(cat,0.5,house,0.5,0)
sub=np.uint8((sum-cat*0.5)*2)
# sub=cv.subtract(sum,house*0.5)#报错
# mul=mask*cat
mul=cv.multiply(cat,mask)#相乘时要把掩膜的像素值转为0,1防止255越界出错
# dev=cat/(noise+1)
dev=cv.divide(cat,noise+1)#相除时要加一防止除0错误
cv.imwrite('sum.png',sum)
cv.imwrite('sub.png',sub)
cv.imwrite('mul.png',mul)
cv.imwrite('dev.png',dev)
import os
import cv2 as cv
import numpy as np
mask=cv.imread('mask.png',0)
mask=(mask/255).astype(np.uint8)
cat=cv.imread('cat.png',0)
house=cv.imread('bg.png',0)
noise=cv.imread('noise .png',0)
sum=cv.addWeighted(cat,0.5,house,0.5,0)
sub=np.uint8((sum-cat*0.5)*2)
# sub=cv.subtract(sum,house*0.5)#报错
# mul=mask*cat
mul=cv.multiply(cat,mask)#需要把掩膜转为0，1像数值否则会因越界255而出错
# dev=cat/(noise+1)
dev=cv.divide(cat,noise+1)#被除数要+1防止除0错误
cv.imwrite('sum.png',sum)
cv.imwrite('sub.png',sub)
cv.imwrite('mul.png',mul)
cv.imwrite('dev.png',dev)
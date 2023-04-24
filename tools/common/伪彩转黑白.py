# import os
# from math import ceil
#
# import cv2
# import numpy
# import selectivesearch
# from tensorboardX import SummaryWriter
#
# def ssw(img):
#     size_min=img.shape[0]**2*0.2
#     size_max=img.shape[0]**2*0.8
#     img_lbl,regions=selectivesearch.selective_search(img,scale=160,sigma=0,min_size=10)
#
#     candidates =set()
#     for r in regions:
#         # 重复的不要
#         if r['rect'] in candidates:
#             continue
#         S=r['rect'][2]*r['rect'][3]
#         # 太小和太大的不要
#         if S > size_max or S < size_min:
#             continue
#
#         # 太不方的不要
#         #if w  > 2*h or h > 2* w :
#         #    continue
#         r=list(r['rect'])
#         r[0]=int(r[0])
#         r[1]=int(r[1])
#         r[2]=int(r[0]+r[2])
#         r[3]=int(r[1]+r[3])
#         candidates.add(tuple(r))
#
#     return candidates
#
# # images=os.listdir('C:\\pythonProject\\study\\binary_SDCN\\train')
# images=os.listdir('./pic')
# writer = SummaryWriter()
# ssw_file=open('train.txt','w')
# for image in images:
#     pic=cv2.imread(os.path.join('./pic', image))
#     pic=cv2.resize(pic,(224,224))
#     proposal=ssw(pic)
#     print(image)
#     print(len(proposal))
#     print(proposal)
#
#     #写入文件
#     a = list(numpy.array(proposal).flat)
#     string = str(image) + " " + " ".join(str(i).replace(',',' ').replace('(',' ').replace(')',' ') for i in a[0]) + '\n'
#     ssw_file.write(string)
#
#     #画框
#     for i,(pro) in enumerate(proposal):
#         pic = cv2.imread(os.path.join("./pic", image))
#         pic = cv2.resize(pic, (224, 224))
#         pic=cv2.rectangle(pic, pro[0:2], pro[2:4], (0, 255, 0), 2)  # 在图片上进行绘制框
#         writer.add_image(image, pic, i, dataformats='HWC')
#         # cv2.imwrite(os.path.join("./result2",image.split('.')[0]+str(pro)+'.jpg'),pic)
#
# # writer.close()
import numpy as np
from PIL import Image

image=Image.open('label.png')
newImg = np.array(image) * 255  # 红色的标签表示灰度值为1,乘以255后都变为255
newImg = newImg.astype(np.uint8)
newImg = Image.fromarray(newImg)
newImg.save('mask.png')
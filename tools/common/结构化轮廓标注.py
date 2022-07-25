# 将专家标注出来的txt文件中的点标到图像上
import os

from PIL import Image
from pylab import *
# import matplotlib.pyplot as plt
import cv2
from matplotlib import pyplot as plt


# 拿到所有内轮廓文件名列表控制程序进行
def icontour__filelist(filepath):
    allfilelist = os.listdir(filepath)
    # 获取所有内轮廓文件
    icontour_filelist = []
    for file in allfilelist:
        if 'icontour' in file:
            icontour_filelist.append(file)
        else:
            continue
    return icontour_filelist


# 处理点数据
def points(file):
    # 读取文件每一行数据
    sourceInLine = file.readlines()  # 内轮廓
    pointspos = []
    for line in sourceInLine:
        temp1 = line.strip('\n')
        temp2 = temp1.split()
        pointspos.append(temp2)
    return pointspos


# 标注点
def draw_points(pointspos, _color):
    for icon in range(0, len(pointspos)):
        x = float(pointspos[icon][0])
        y = float(pointspos[icon][1])
        plt.plot(x, y, '.', markersize=0.5, color=_color)


if __name__ == '__main__':
    # 拿到存放标注文件的文件夹路径
    filepath = 'E:/graduate_study/work/full_dataset/TrainingSet_contours'
    # 拿到放图片的文件夹的路径
    imagepath = 'E:/graduate_study/work/full_dataset/TrainingSet_images'
    # 标注图像的存放位置
    savepath = 'E:/graduate_study/work/full_dataset/TrainingSet_images_contours'
    icontour_filelist = icontour__filelist(filepath)

    for icontour_file in icontour_filelist:
        # 根据内文件拼接出外文件路径
        ocontour_file_path = os.path.join(filepath, icontour_file.replace('icontour', 'ocontour'))
        # 拼接出内文件路径
        icontour_file_path = os.path.join(filepath, icontour_file)
        # 拼接出要标注的图像的路径
        image_contour_path = os.path.join(imagepath, icontour_file[0:8] + '.png')
        # 打开内轮廓标注文件
        iconfile = open(icontour_file_path)
        # 打开外轮廓标注文件
        oconfile = open(ocontour_file_path)
        # 打开被标注图片
        image_open=imread(image_contour_path)
        img = array(image_open)
        # 设置大小
        plt.figure(figsize=(image_open.shape[1]/100,image_open.shape[0]/100 ), dpi=100)
        plt.axes([0., 0., 1., 1.])
        # 设置以灰度图片方式打开
        plt.imshow(img, cmap='gray')
        icon_pointspos = points(iconfile)  # 获取内轮廓点
        ocon_pointspos = points(oconfile)  # 获取外轮廓点
        draw_points(icon_pointspos, 'g')  # 画内轮廓
        draw_points(ocon_pointspos, 'r')  # 画外轮廓
        # 关闭坐标轴
        plt.axis('off')
        # 拼接出标注图的位置
        save_image_path = os.path.join(savepath, icontour_file[0:8] + '.png')
        # 保存图片,除白边
        plt.savefig(save_image_path, dpi=100, bbox_inches='tight', pad_inches=0)
        plt.clf()
        plt.cla()
        plt.close()
#__________________________________________原始程序__________________________________________________
# #拿到存放标注文件的文件夹路径
# filepath='E:/graduate_study/work/full_dataset/TrainingSet_contours'
# #拿到放图片的文件夹的路径
# imagepath='E:/graduate_study/work/full_dataset/TrainingSet_images'
# #标注图像的存放位置
# savepath='E:/graduate_study/work/full_dataset/TrainingSet_images_contours'
# #拿到所有文件的文件名
# allfilelist=os.listdir(filepath)
# # 获取所有内轮廓文件
# icontour_filelist=[]
# for file in allfilelist:
#     if 'icontour' in file:
#         icontour_filelist.append(file)
#     else:
#         continue
# #遍历所有内文件
# for icontour_file in icontour_filelist:
#     # 根据内文件拼接出外文件路径
#     ocontour_file_path=os.path.join(filepath,icontour_file.replace('icontour','ocontour'))
#     #拼接出内文件路径
#     icontour_file_path = os.path.join(filepath,icontour_file)
#     #拼接出要标注的图像的路径
#     image_contour_path=os.path.join(imagepath,icontour_file[0:8]+'.png')
#     #打开内轮廓标注文件
#     iconfile = open(icontour_file_path)
#     # 打开外轮廓标注文件
#     oconfile = open(ocontour_file_path)
#     #打开被标注图片
#     img = array(imread(image_contour_path))
#     #设置以灰度图片方式打开
#     plt.imshow(img, cmap='gray')
#     #读取文件每一行数据
#     icon_sourceInLine = iconfile.readlines()#内轮廓
#     ocon_sourceInLine = oconfile.readlines()#外轮廓
#     icon_pointspos = []  # 内轮廓点
#     ocon_pointspos = []  # 外轮廓点
#     #处理内轮廓点数据
#     for icon_line in icon_sourceInLine:
#         temp1 = icon_line.strip('\n')
#         temp2 = temp1.split()
#         icon_pointspos.append(temp2)
#     #处理外轮廓点
#     for ocon_line in ocon_sourceInLine:
#         ref1 = ocon_line.strip('\n')
#         ref2 = ref1.split()
#         ocon_pointspos.append(ref2)
#     #画点
#     # 每行都是横坐标，纵坐标
#     # 标内轮廓点
#     for icon in range(0, len(icon_pointspos)):
#         icon_x = float(icon_pointspos[icon][0])
#         icon_y = float(icon_pointspos[icon][1])
#         plt.plot(icon_x, icon_y, '.', markersize=1, color='g')
#     #标外轮廓的点
#     for ocon in range(0, len(ocon_pointspos)):
#         ocon_x = float(ocon_pointspos[ocon][0])
#         ocon_y = float(ocon_pointspos[ocon][1])
#         plt.plot(ocon_x, ocon_y, '.', markersize=1, color='r')
#     #关闭坐标轴
#     plt.axis('off')
#     #拼接出标注图的位置
#     save_image_path=os.path.join(savepath,icontour_file[0:8]+'.png')
#     #保存图片,除白边
#     plt.savefig(save_image_path, dpi=200, bbox_inches='tight', pad_inches=0)
#     plt.clf()

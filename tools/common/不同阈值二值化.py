from PIL import Image
import os

"""
    从0-255每个阈值都二值化一次
"""
path = "last.png"# 原始路径
save_path = '/tools/points_to_binary_images/last2.png'  # 保存路径
img = Image.open(path)  # 打开图片
# 模式L”为灰色图像，它的每个像素用8个bit表示，0表示黑，255表示白，其他数字表示不同的灰度。
Img = img.convert('L')

for a in range (0,256):
    # 自定义灰度界限，小于这个值为黑色，大于这个值为白色
    threshold = a

    table = []
    #
    for i in range(256):
        if i == threshold:
            table.append(255)
        else:
            table.append(0)
    # 图片二值化 使用table来设置二值化的规则
    photo = Img.point(table, '1')#1表示不进行加强，若为1.2则表示对图像进行20%的加强
    #拼接图片保存路径
    last_path=os.path.join(save_path,str(a)+'.png')
    photo.convert("L").save(last_path)#将最终的图片转为8位深的灰度图保存

"""
    按指定的一个阈值二值化
"""
##一个阈值
# # 自定义灰度界限，小于这个值为黑色，大于这个值为白色
# threshold = 255
#
# table = []
# #
# for i in range(256):
#     if i >= threshold:
#         table.append(255)
#     else:
#         table.append(0)
#
# # 图片二值化 使用table来设置二值化的规则
# photo = Img.point(table, '1')  # 1表示不进行加强，若为1.2则表示对图像进行20%的加强
# photo.convert("L").save(save_path)  # 将最终的图片转为8位深的灰度图保存
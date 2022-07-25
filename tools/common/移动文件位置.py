import os
import shutil

# # 3层文件夹
# # 想要移动文件所在的根目录
# rootdir = "E:/graduate_study/work/full_dataset/TrainingSet"
# # 目标路径
# des_path = "E:/graduate_study/work/full_dataset/TrainingSet_contours"
# # 获取目录下文件名清单
# list = os.listdir(rootdir)  #
# # print(list)
#
# # 移动图片到指定文件夹
# for i in range(0, len(list)):  # 遍历目录下的所有文件夹    遍历patient目录
#     path = os.path.join(rootdir, list[i])
#     objectlist = os.listdir(path)  # P01contours-manual、P01dicom、P01list.txt
#     for item in objectlist:
#         if 'manual' in item:  # 获取标注文件
#             lastpath = os.path.join(path, item)  # 拼接txt路径
#             lastlist = os.listdir(lastpath)  # 获取要移动的文件
#             for objectfile in lastlist:
#                 if 'ocontour' in objectfile:
#                     filepath = os.path.join(lastpath, objectfile)
#                     movepath = os.path.join(des_path,'ocontour',objectfile)
#                     shutil.copy(filepath, movepath)#复制文件
#                 #shutil.move(filepath, movepath)#剪切文件
#         else:
#             continue
# print("OK")
#
#一层文件夹
# 想要移动文件所在的根目录
rootdir = "E:/graduate_study/work/full_dataset/Test1SetContours"
# 目标路径
des_path = "E:/graduate_study/work/full_dataset/Test1Set_contours"
#获取一级文件夹
firstlist=os.listdir(rootdir)

for item1 in firstlist:#遍历一级文件夹
    secpath=os.path.join(rootdir,item1)#拼接出以及文件夹的路径
    seclist=os.listdir(secpath)#获取要移动的文件路径
    for item2 in seclist:
        if 'icontour' in item2:
            filepath = os.path.join(secpath, item2)
            copypath = os.path.join(des_path, 'icontour',item2)  # 拼接出目标路径
            shutil.copy(filepath, copypath)
print("OK")

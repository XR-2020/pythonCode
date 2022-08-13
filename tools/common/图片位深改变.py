from PIL import Image
import os


# #改变图片的样式，灰度L，RGB，黑白1
# path = "E:/graduate_study/work/full_dataset/TrainingSet_labels"# 原始路径
# save_path = path# 保存路径
# all_images = os.listdir(path)
# print(all_images)
#
# for image in all_images:
#     print(image)
#     image_path = os.path.join(path, image)
#     img = Image.open(image_path)  # 打开图片
#     print(img.format, img.size, img.mode)#打印出原图格式
#     img = img.convert("L")  # 4通道转化为灰度L通道
#     img.save(os.path.join(path, image))

#操作单张图
img = Image.open('mask.png')  # 打开图片
print(img.format, img.size, img.mode)#打印出原图格式
img = img.convert("RGB")  # 4通道转化为灰度L通道
img.save('mask.png')
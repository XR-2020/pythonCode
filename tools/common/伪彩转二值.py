from PIL import Image
import os

path = "E:/python/MyUnet/a/"# 原始路径
save_path = 'E:/python/MyUnet/a/'# 保存路径
all_images = os.listdir(path)

for image in all_images:
    image_path = os.path.join(path, image)
    img = Image.open(image_path)  # 打开图片
    # 模式L”为灰色图像，它的每个像素用8个bit表示，0表示黑，255表示白，其他数字表示不同的灰度。
    Img = img.convert('L')
    # Img.save("L.png")

    # 自定义灰度界限，小于这个值为黑色，大于这个值为白色
    threshold = 1

    table = []
    #
    for i in range(256):
        if i >= threshold:
            table.append(255)
        else:
            table.append(0)

    # 图片二值化 使用table来设置二值化的规则
    photo = Img.point(table, '1')#1表示不进行加强，若为1.2则表示对图像进行20%的加强
    photo.convert("L").save(save_path + image)#将最终的图片转为8位深的灰度图保存

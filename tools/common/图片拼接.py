# coding=utf-8
from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np


def jigsaw(imgs, direction="horizontal", gap=0):
    imgs = [Image.fromarray(img) for img in imgs]
    w, h = imgs[0].size
    if direction == "horizontal":
        result = Image.new(imgs[0].mode, ((w + gap) * len(imgs) - gap, h))
        for i, img in enumerate(imgs):
            result.paste(img, box=((w + gap) * i, 0))
    elif direction == "vertical":
        result = Image.new(imgs[0].mode, (w, (h + gap) * len(imgs) - gap))
        for i, img in enumerate(imgs):
            result.paste(img, box=(0, (h + gap) * i))
    else:
        raise ValueError("The direction parameter has only two options: horizontal and vertical")
    return np.array(result)


if __name__ == '__main__':
    metails_list = []
    pre_list = []
    exp_list = []
    matails_path = 'E:/python/picture/test/'
    pre_path = 'E:/python/picture/mymask/'
    exp_path = 'E:/python/picture/mask_RV/'
    # 拼接原图
    for i in range(0, 16):
        metails_list.append(cv2.imread(matails_path + str(i) + '.png'))
    img = jigsaw(metails_list)
    cv2.imwrite("matails.png", img)

    #     拼接预测
    for i in range(0, 16):
        pre_list.append(cv2.imread(pre_path + str(i) + '.png'))
    img = jigsaw(pre_list)
    cv2.imwrite("predict.png", img)
    # 拼接专家图
    for i in range(1, 17):
        exp_list.append(cv2.imread(exp_path + str(i) + '.png'))
    img = jigsaw(exp_list)
    cv2.imwrite("exptor.png", img)

    # 纵向合并
    img1 = Image.open('matails.png')
    img2 = Image.open('predict.png')
    img3 = Image.open('exptor.png')
    size1, size2, size3 = img1.size, img2.size, img3.size
    joint = Image.new("RGB", (size1[0], size1[1] + size2[1] + size3[1]))
    loc1, loc2, loc3 = (0, 0), (0, size1[1]), (0, size1[1] + size2[1])
    joint.paste(img1, loc1)
    joint.paste(img2, loc2)
    joint.paste(img3, loc3)
    joint.save('vertical.png')

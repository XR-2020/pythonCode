import os
from shutil import copy, rmtree
import random
from PIL import Image
from PIL import ImageEnhance

import cv2
import numpy as np

def flip(img):  # 水平翻转图像
    # img = Image.open(os.path.join(root_path, img_name))
    filp_img = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    filp_img.save('flip.jpg')


def vflip(img):  # 竖直翻转图像
    filp_img = img.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
    filp_img.save('vflip.jpg')

def rotation_90(img):
    rotation_img = img.rotate(280)  # 旋转角度
    rotation_img.save('rotation280.jpg')
    # rotation_img = img.rotate(90)  # 旋转角度
    # rotation_img.save('rotation2.jpg')


def randomColor(image):  # 随机颜色
    random_factor = np.random.randint(0, 31) / 10.  # 随机因子
    color_image = ImageEnhance.Color(image).enhance(random_factor)  # 调整图像的饱和度
    random_factor = np.random.randint(10, 21) / 10.  # 随机因子
    brightness_image = ImageEnhance.Brightness(color_image).enhance(random_factor)  # 调整图像的亮度
    random_factor = np.random.randint(10, 21) / 10.  # 随机因子
    contrast_image = ImageEnhance.Contrast(brightness_image).enhance(random_factor)  # 调整图像对比度
    random_factor = np.random.randint(0, 31) / 10.  # 随机因子
    result=ImageEnhance.Sharpness(contrast_image).enhance(random_factor)  # 调整图像锐度
    result.save('randomColor.jpg')


def contrastEnhancement(image):  # 对比度增强
    enh_con = ImageEnhance.Contrast(image)
    contrast = 2.0
    image_contrasted = enh_con.enhance(contrast)
    image_contrasted.save('contrastEnhancement.jpg')


def brightnessEnhancement(image):  # 亮度增强
    enh_bri = ImageEnhance.Brightness(image)
    brightness = 1.5
    image_brightened = enh_bri.enhance(brightness)
    image_brightened.save('brightnessEnhancement.jpg')
    return image_brightened



def colorEnhancement(img):  # 颜色增强
    enh_col = ImageEnhance.Color(img)
    color = 2.0
    image_colored = enh_col.enhance(color)
    image_colored.save('colorEnhancement.jpg')
    return image_colored

def crop(img):
    region = img.crop((50, 45 ,140, 140))  ## 0,0表示要裁剪的位置的左上角坐标，50,50表示右下角。
    region.save('crop.png')  ## 将裁剪下来的图片保存到 举例.png

def main():

    image=Image.open('61.png')
    # flip(image)
    # vflip(image)
    #rotation_90(image)
    # crop(image)
    # contrastEnhancement(colorEnhancement(brightnessEnhancement(image)))
    colorEnhancement(image)
    contrastEnhancement(image)
   # randomColor(image)
    print("processing done!")


if __name__ == '__main__':
    main()
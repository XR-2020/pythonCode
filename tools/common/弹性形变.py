# -*- coding:utf-8 -*-
import cv2
import numpy as np
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.interpolation import map_coordinates


def elastic_transform(image, alpha, sigma,
                      alpha_affine, random_state=None):

    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape
    shape_size = shape[:2]
    # Random affine
    center_square = np.float32(shape_size) // 2
    square_size = min(shape_size) // 3
    # pts1: 仿射变换前的点(3个点)
    pts1 = np.float32([center_square + square_size,
                       [center_square[0] + square_size,
                        center_square[1] - square_size],
                       center_square - square_size])
    # pts2: 仿射变换后的点
    pts2 = pts1 + random_state.uniform(-alpha_affine, alpha_affine,
                                       size=pts1.shape).astype(np.float32)
    # 仿射变换矩阵
    M = cv2.getAffineTransform(pts1, pts2)
    # 对image进行仿射变换.
    imageB = cv2.warpAffine(image, M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101)

    # generate random displacement fields
    # random_state.rand(*shape)会产生一个和shape一样打的服从[0,1]均匀分布的矩阵
    # *2-1是为了将分布平移到[-1, 1]的区间, alpha是控制变形强度的变形因子
    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    # generate meshgrid
    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    # x+dx,y+dy
    indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))
    # bilinear interpolation
    imageC = map_coordinates(imageB, indices, order=1, mode='constant').reshape(shape)

    return imageC


if __name__ == '__main__':
    img_path = '61.png'
    imageA = cv2.imread(img_path)
    img_show=cv2.cvtColor(imageA.copy(), cv2.COLOR_BGR2GRAY)
    imageA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
    # Apply elastic transform on image
    imageC = elastic_transform(imageA, imageA.shape[1] * -3,
                                   imageA.shape[1] * 0.08,
                                   imageA.shape[1] * 0.08)
    imageD = elastic_transform(imageA, imageA.shape[1] * 0,
                               imageA.shape[1] * 0.08,
                               imageA.shape[1] * 0.08)
    imageE = elastic_transform(imageA, imageA.shape[1] * 3,
                               imageA.shape[1] * 0.08,
                               imageA.shape[1] * 0.08)
    result1=np.hstack((img_show,imageC))
    result2=np.hstack((img_show,imageD))
    result3=np.hstack((img_show,imageE))
    result4=np.vstack((result1,result2))
    result=np.vstack((result4,result3))
    # cv2.namedWindow("img_a", 0)
    # cv2.imshow("img_a", img_show)
    # #cv2.namedWindow("img_c", 0)
    cv2.imshow("img", result)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
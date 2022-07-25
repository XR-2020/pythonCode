#将专家的点坐标转为黑白图
import os
import cv2 as cv
from PIL import Image
from pylab import *
from matplotlib import pyplot as plt

def chang_black(image_path,save_path):
    """将图片全部转为黑色
    :param image_path: 要转为黑色图像的图片地址    xxx.png
    :param save_path: 黑色图像转换完成后要保存的地址 xxx.png
    :return: 无
    """
    img = Image.open(image_path)  # 打开图片
    # 模式L”为灰色图像，它的每个像素用8个bit表示，0表示黑，255表示白，其他数字表示不同的灰度。
    Img = img.convert('L')
    # Img.save("L.png")
    # 自定义灰度界限，小于这个值为黑色，大于这个值为白色
    threshold = 0
    table = []
    for i in range(256):
        if i < threshold:
            table.append(255)
        else:
            table.append(0)
    # 图片二值化 使用table来设置二值化的规则
    photo = Img.point(table, '1')  # 1表示不进行加强，若为1.2则表示对图像进行20%的加强
    photo.convert("L").save(save_path)  # 将最终的图片转为8位深的灰度图保存

#画轮廓
def draw_contour(filepath,imagepath,savepath):
    """

    :param filepath: 专家标注的轮廓的地址 xxxx.txt
    :param imagepath: 要进行轮廓描绘的图像的地址    xxxxx.png
    :param savepath: 描完轮廓后图像保存的地址  xxxx.png
    :return:
    """
    # 打开内轮廓标注文件
    file = open(filepath)
    # 打开被标注图片
    img = array(imread(imagepath))
    # 设置以灰度图片方式打开
    image_open = imread(imagepath)
    img = array(image_open)
    # 设置大小防止图片大小改变
    plt.figure(figsize=(image_open.shape[1] / 100, image_open.shape[0] / 100), dpi=100)
    plt.axes([0., 0., 1., 1.])
    plt.imshow(img, cmap='gray')
    # 读取文件每一行数据
    sourceInLine = file.readlines()  # 内轮廓
    x=[]#轮廓x坐标
    y=[]#轮廓y坐标
    # 处理内轮廓点数据
    for icon_line in sourceInLine:
        temp1 = icon_line.strip('\n')
        temp2 = temp1.split()
        #将（x,y）中的x加入到x的数组中
        x.append(float(temp2[0]))
        # 将（x,y）中的y加入到y的数组中
        y.append(float(temp2[1]))
    for icon in range(0, len(x)):
        icon_x = x[icon]
        icon_y = y[icon]
        plt.plot(icon_x, icon_y, '.', markersize=0.5, color='w')
    #连点成线防止产生多不闭合区域而造成后期填充错误
    plt.plot(x,y, linewidth=1, color='w')
    # 关闭坐标轴
    plt.axis('off')
    # 保存图片,除白边
    plt.savefig(savepath, dpi=100, bbox_inches='tight', pad_inches=0)
    plt.clf()
    plt.close()

#轮廓填色
def fill_color(image_path,save_path):
    """
    :param image_path: 要填充轮廓的图片的地址 xxx.png
    :param save_path: 最终生成的标签要保存的地方 xxx.png
    :return:
    """
    # 读取图像
    img_source = cv.imread(image_path)  # 原图（三通道）
    # 将图像转为黑白图
    img = cv.cvtColor(img_source, cv.COLOR_BGR2GRAY)
    # 检测轮廓
    contours, hierarchy = cv.findContours(img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    # # 描绘出检测到的轮廓
    # cv.drawContours(img_source, contours, -1, (0, 0, 0), -1)
    # 给轮廓填充颜色
    cv.fillPoly(img_source, [contours[1]], (255, 255, 255))  # 填充内部
    # cv.imshow("result", img_source)
    # 转为灰度图
    save_image = img = cv.cvtColor(img_source, cv.COLOR_BGR2GRAY)
    # 保存图片
    cv.imwrite(save_path, save_image)
    # 销毁窗口
    cv.destroyAllWindows()


if __name__ == '__main__':
    """
        1.换不同的数据集时要改4个地方images_path、contourfiles_path、contour_images_path、labels_path
        2.换同一个数据集不同轮廓时时要改3个地方contourfiles_path、contour_images_path、labels_path
    """

    #原始图像位置
    images_path='E:/graduate_study/work/full_dataset/Test2Set_images'

    #轮廓txt文件的位置
    contourfiles_path='E:/graduate_study/work/full_dataset/Test2Set_contours/ocontour'

    #二值化全黑图像保存位置
    binary_images_path= '/tools/points_to_binary_images/ref'

    #描完轮廓的图片放的位置
    contour_images_path='E:/graduate_study/work/full_dataset/Test2Set_labels/contour/ocontour'

    #标签的位置
    labels_path='E:/graduate_study/work/full_dataset/Test2Set_labels/labels/ocontour'

    #获取需要转换成黑图的图片的文件名
    image_name=[]
    black_picture_path=[]
    dir=os.listdir(contourfiles_path)
    #读取轮廓文件列表   xxxx.txt
    for image in dir:
        image_name.append(image[0:8]+'.png')
        black_picture_path.append(os.path.join(binary_images_path,image[0:8]+'.png'))

    # 将有标注的图片转成黑色图
    for i in range(0, len(image_name)):
        chang_black(os.path.join(images_path, image_name[i]), black_picture_path[i])
    print("黑色图OK")

    # 获取轮廓标注文件列表
    filelist = os.listdir(contourfiles_path)
    # 一个个处理轮廓文件，画轮廓
    for file in filelist:  # 每个标注文件
        draw_contour(os.path.join(contourfiles_path, file), os.path.join(binary_images_path, file[0:8] + '.png'),
                     os.path.join(contour_images_path, file[0:8] + '.png'))
    print("轮廓文件OK")

    #依次处理每个黑白图像，轮廓内填充颜色
    for binary_image in image_name:
        fill_color(os.path.join(contour_images_path,binary_image),os.path.join(labels_path,binary_image))
    print("OK")
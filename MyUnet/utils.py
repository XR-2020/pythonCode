from PIL import Image

#归一化图片统一转成256*256
def keep_image_size_open(path, size=(256, 256)):
    img = Image.open(path)
    temp = max(img.size)
    """
        Image.new(mode, size, color=0)
            mode:模式，通常用"RGB"这种模式，如果需要采用其他格式，可以参考博文：PIL的mode参数
            size：生成的图像大小
            color：生成图像的颜色，默认为0，即黑色
    """
    mask = Image.new('RGB', (temp, temp), (0, 0, 0))
    # 将一张图粘贴到另一张图像上,paste(image,box)变量box是一个给定左上角坐标的2元组
    mask.paste(img, (0, 0))
    mask = mask.resize(size)
    return mask

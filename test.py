import math

import cv2
import selectivesearch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# 第二步：执行搜索工具,展示搜索结果
image2 = "000005.jpg"

# 用cv2读取图片
img = cv2.imread(image2)

# # 白底黑字图 改为黑底白字图
# img = 255 - img

# selectivesearch 调用selectivesearch函数 对图片目标进行搜索
# img_lbl, regions = selectivesearch.selective_search(img, scale=500, sigma=0.7, min_size=20)
def ssw(img,scale=500,sigma=0.7,min_size=20):
    img_lbl,regions=selectivesearch.selective_search(img,scale=scale,sigma=sigma,min_size=min_size)
    candidates =set()
    for r in regions:
        # 重复的不要
        if r['rect'] in candidates:
            continue
        # # 太大的不要
        if r['size'] > 2000 or r['size']<5:
            continue
        #x, y, w, h = r['rect']
        # 太不方的不要
        #if w  > 2*h or h > 2* w :
        #    continue
        candidates.add(r['rect'])
    return list(candidates)

regions = ssw(img, scale=500, sigma=0.7, min_size=20)
mapping=[]
for ele in regions:
    print("__________________________________________________-")
    # ceil向上取整，floor向下取整
    mapping.append((math.floor(ele[0] / 16) + 1, math.floor(ele[1] / 16) + 1,
                    math.ceil((ele[0] + ele[2]) / 16) - 1 - (math.floor(ele[0] / 16) + 1),
                    math.ceil((ele[1] + ele[3]) / 16) - 1 - (math.floor(ele[1] / 16) + 1)))
    print(ele)
    print("************")
    print((math.floor(ele[0] / 16) + 1, math.floor(ele[1] / 16) + 1,
                    math.ceil((ele[0] + ele[2]) / 16) - 1 - (math.floor(ele[0] / 16) + 1),
                    math.ceil((ele[1] + ele[3]) / 16) - 1 - (math.floor(ele[1] / 16) + 1)))
    print("************")
    print((math.floor(ele[0] / 16) + 1, math.floor(ele[1] / 16) + 1,
                                    math.ceil(ele[2] / 16),
                                    math.ceil(ele[3] / 16)))
    # mapping.append((math.floor(ele[0] / 16) + 1, math.floor(ele[1] / 16) + 1,
    #                                 math.ceil(ele[2] / 16),
    #                                 math.ceil(ele[3] / 16)))
mapping = list(set(mapping))

# # 接下来我们把窗口和图像打印出来，对它有个直观认识
# fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
# ax.imshow(img)
# for reg in mapping:
#     x, y, w, h = reg
#     rect = mpatches.Rectangle((x, y), w, h, fill=False, edgecolor='red', linewidth=1)
#     # if reg['size']<500 or reg['size']>1000:
#     #     continue
#     ax.add_patch(rect)
#
# plt.show()
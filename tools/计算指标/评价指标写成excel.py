import xlwt
import os

import cv2
import operator
import matplotlib.pyplot as plt
import torch


def dice_coef(output, target):  # output为预测结果 target为真实结果
    smooth = 1e-5  # 防止0除

    intersection = (output * target).sum()

    return (2. * intersection + smooth) / \
           (output.sum() + target.sum() + smooth)
def iou_score(output, target):
    smooth = 1e-5
    #0,1化矩阵
    output_ = output > 0.5
    target_ = target > 0.5
    intersection = (output_ & target_).sum()
    union = (output_ | target_).sum()

    return (intersection + smooth) / (union + smooth)
def ppv_score(output, target):
    smooth = 1e-5

    intersection = (output * target).sum()

    return (intersection + smooth) / \
           (output.sum() + smooth)
def sensitivity(output, target):
    smooth = 1e-5


    intersection = (output * target).sum()

    return (intersection + smooth) / \
           (target.sum() + smooth)

def img_input(myMark_path, result_path):
    myMark = cv2.imread(myMark_path)  # 使用 cv2 来打开图片
    result = cv2.imread(result_path)  # 使用 cv2 来打开图片
    return (myMark, result)


if __name__ == '__main__':
    # 创建新的workbook（其实就是创建新的excel）
    workbook = xlwt.Workbook(encoding='ascii')

    # 创建新的sheet表
    worksheet = workbook.add_sheet("icontour")

    myMark_path1 = "E:/python/cnn_demo/result/test1/ocontour"
    result_path1 = "E:/graduate_study/work/full_dataset/Test1Set/Test1Set_labels/ocontour"
    filelist = os.listdir(result_path1)
    dicelist = []
    ioulist=[]
    ppvlist=[]
    senlist=[]
    namelist=[]
    for item in filelist:
        namelist.append(item)
        myMark_path = os.path.join(os.path.abspath(myMark_path1), item)
        result_path = os.path.join(os.path.abspath(result_path1), item)
        print(myMark_path)
        myMark, result = img_input(myMark_path, result_path)

        dice = dice_coef(result, myMark)
        iou=iou_score(result,myMark)
        ppv=ppv_score(result,myMark)
        sen=sensitivity(result,myMark)

        dicelist.append(dice)
        ioulist.append(iou)
        ppvlist.append(ppv)
        senlist.append(sen)

    worksheet.write(0, 1, 'Dice')
    worksheet.write(0, 2, 'IOU')
    worksheet.write(0, 3,'PPV')
    worksheet.write(0, 4,'Sensitivity')
    # 往表格写入内容
    for i in range(len(namelist)):
        worksheet.write(i+1,0, namelist[i])

    for i in range(len(namelist)):
        worksheet.write(i+1,1, dicelist[i])
        worksheet.write(i+1, 2, ioulist[i])
        worksheet.write(i+1, 3, ppvlist[i])
        worksheet.write(i+1, 4, senlist[i])

    # 保存
    workbook.save("test1_ooccutour.xls")
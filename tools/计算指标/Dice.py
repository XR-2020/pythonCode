# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
# from filecmp import cmp
import os

import cv2
import operator
import matplotlib.pyplot as plt
import torch


# dice
def dice_coef(output, target):  # output为预测结果 target为真实结果
    smooth = 1e-5  # 防止0除

    intersection = (output * target).sum()

    return (2. * intersection + smooth) / \
           (output.sum() + target.sum() + smooth)


def img_input(myMark_path, result_path):
    myMark = cv2.imread(myMark_path)  # 使用 cv2 来打开图片
    result = cv2.imread(result_path)  # 使用 cv2 来打开图片
    return (myMark, result)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # print_hi('PyCharm')
    myMark_path1 = "E:/python/MyUnet/test_data/label/"
    result_path1 = "E:/python/mask_RV/"
    filelist=os.listdir(result_path1)
    i = 1
    dicelist = []
    namelist=[]
    for item in filelist:
        namelist.append(item)
        myMark_path = os.path.join(os.path.abspath(myMark_path1), item)
        result_path = os.path.join(os.path.abspath(result_path1), item)
        myMark, result = img_input(myMark_path, result_path)
        dice = dice_coef(result, myMark)
        dicelist.append(round(dice, 4))
        print(dice)
        i = i + 1
        fun3_2_2('Dice',namelist,dice,1,'test1')
    #
    # x_values = range(1, 17)
    # y_values = dicelist
    # '''
    # plt.scatter()
    # #x:横坐标 y:纵坐标 s:点的尺寸
    # '''
    # plt.scatter(x_values, y_values, s=80)
    #
    # # 设置图表标题并给坐标轴加上标签
    # plt.title('Dice', fontsize=24)
    # plt.xlabel('Image sequence number', fontsize=14)
    # plt.ylabel('dice', fontsize=14)
    # # 显示网格
    # plt.grid(True)
    # # 设置刻度标记的大小
    # plt.tick_params(axis='both', which='major', labelsize=15)
    #
    # # 显示指标数值大小
    # for i in range(len(y_values)):
    #     plt.text(x_values[i], y_values[i], '%.3f' % (dicelist[i]), ha='center', va='bottom', fontsize=8)
    #
    # # plt.show()
    # #
    # # dicesum = 0
    # # num = 0
    # # for num in range(0, 16):
    # #     dicesum = dicesum + dicelist[num]
    # # print("dicesum:", dicesum, "num:", num + 1)
    # # avg = dicesum / 16
    # # print("avg:", avg)
    # # c = max(x_values)
    # # d = (max(y_values)+min(y_values))/2
    # # plt.text(c, d, 'dicesum=%.2f' % dicesum, ha='left', va='bottom', fontsize=15)
    # # plt.text(c, d+1, 'avg=%.2f' % avg, ha='left', va='bottom', fontsize=15)
    # # 自动保存结果图片
    # plt.savefig('Dice.jpg', bbox_inches='tight')

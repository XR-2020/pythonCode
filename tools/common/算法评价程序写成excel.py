import xlwt
import os

# 创建新的workbook（其实就是创建新的excel）
workbook = xlwt.Workbook(encoding='ascii')

# 创建新的sheet表
worksheet = workbook.add_sheet("contour")


worksheet.write(0, 1, 'Dice')
worksheet.write(0, 2, 'Hausdorff')

data = open('D:/full_data/Test1AutomaticContours_ImageResults.txt')
sourceInLine = data.readlines()  # 内轮廓
name = []
dice=[]
hau=[]
for line in sourceInLine:
    temp1 = line.strip('\n')
    temp2 = temp1.split()
    name.append(temp2[0])
    dice.append(temp2[1])
    hau.append([temp2[2]])

for i in range(len(name)):
    worksheet.write(i + 1, 0, name[i])
    worksheet.write(i + 1, 1, dice[i])
    worksheet.write(i + 1, 2, hau[i])

# 保存
workbook.save("test1contour.xls")
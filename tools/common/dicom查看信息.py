import pydicom
import os


def loadFileInformation(filename):
    information = {}

    ds = pydicom.read_file(filename)  # 文件路径

    information['PatientID'] = ds.PatientID#患者的ID

    information['PatientName'] = ds.PatientName#患者姓名

    information['PatientBirthDate'] = ds.PatientBirthDate#患者出生日期

    information['PatientSex'] = ds.PatientSex#患者性别

    information['StudyID'] = ds.StudyID#检查ID

    information['StudyDate'] = ds.StudyDate#检查日期

    information['StudyTime'] = ds.StudyTime#检查时间

    information['InstitutionName'] = ds.InstitutionName#机构名称

    information['Manufacturer'] = ds.Manufacturer#设备制造商

    information['StudyDescription'] = ds.StudyDescription#检查项目描述

    print(information)


if __name__ == '__main__':
    path = "E:/graduate_study/metails/P01dicom/P01-0000.dcm"
    loadFileInformation(path)

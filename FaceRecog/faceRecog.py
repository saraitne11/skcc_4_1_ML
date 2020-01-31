from PIL import Image
import numpy as np
import os
import csv


def readData():
    dataList = []
    imagePath = "../_Data/1. 얼굴사진분류 데이터/image_v2/face_images_128x128/"
    trainPath = "../_Data/1. 얼굴사진분류 데이터/ml_8_faceclassifier_train.csv"

    f = open(trainPath, 'r', encoding="utf-8")
    reader = list(csv.reader(f))
    reader.pop(0)
    f.close()
    for line in reader:
        tempList = []
        tempList.append(line[0])
        tempList.append(line[1])
        img = Image.open(imagePath + tempList[0], "r")
        tempList.append(np.array(img))
        dataList.append(tempList)
        print(tempList)

    # 파일이름, 레이블, 이미지데이터
    return dataList

def main():
    data = readData()

if __name__ == "__main__":
    main()

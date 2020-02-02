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


# TODO
# ReadData 클래스로 만들기
# example
# class DataSet
# def __init__(이미지파일 폴더, csv파일 경로):
#       self.x = 이미지 데이터    [데이터 개수, 128, 128, 3]
#       self.y = 이미지 레이블    [데이터 개수]    (int)
#       self.file_name = 파일 이름  [데이터 개수] (문자열)
#       self.num_data = 데이터 개수
#
# def random_batch(batch_size):
#       x = 랜덤한 batch_size개의 이미지 데이터    [batch_size, 128, 128, 3]
#       y = 랜덤한 batch_size개의 레이블          [batch_size]
#           당연히 이미지 데이터와 레이블이 매칭 되어야함
#       return x, y
#
# def sequential_batch(batch_size):

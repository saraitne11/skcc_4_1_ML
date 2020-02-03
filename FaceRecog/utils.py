from PIL import Image
import numpy as np
import os
import csv
import random


class DataSet:
    def __init__(self):
        imagePath = "../_Data/1. 얼굴사진분류 데이터/image_v2/face_images_128x128/"
        csvPath = "../_Data/1. 얼굴사진분류 데이터/ml_8_faceclassifier_train.csv"

        self.imageData = None
        self.imageLable = None
        self.fileName = []
        self.numData = None
        self.currentIdx = 0

        f = open(csvPath, 'r', encoding="utf-8")
        reader = list(csv.reader(f))
        reader.pop(0)
        f.close()
        self.numData = len(reader)

        self.imageData = np.zeros([self.numData, 128, 128, 3], dtype=np.uint8)
        self.imageLable = np.zeros([self.numData], dtype=np.uint8)

        for i, line in enumerate(reader):
            try:
                tempImg = Image.open(imagePath + line[0], "r")
                self.fileName.append(line[0])
                self.imageData[i, :] = np.array(tempImg)
                self.imageLable[i] = np.array(line[1])
            except ValueError:
                print("face_" + str(i) + ".png 파일은 RGB가 아니고 RGBA 입니다")

    def random_batch(self, batch_size):
        idx = random.sample(list(range(0, self.numData)), batch_size)
        x = self.imageData[idx, :]
        y = self.imageLable[idx]
        return x, y

    def sequential_batch(self, batch_size):
        if self.currentIdx + batch_size <= self.numData:
            x = self.imageData[self.currentIdx:][:self.currentIdx + batch_size]
            y = self.imageLable[self.currentIdx:][:self.currentIdx + batch_size]
            self.currentIdx += batch_size
            return x, y, False, None
        else:
            x = self.imageData[self.currentIdx:]
            y = self.imageLable[self.currentIdx:]
            return x, y, True, "faceRecogResut.csv"


def main():
    datee = DataSet()


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
#       모델 학습용, 에포크
#       x = 랜덤한 batch_size개의 이미지 데이터    [batch_size, 128, 128, 3]
#       y = 랜덤한 batch_size개의 레이블          [batch_size]
#           이미지 데이터와 레이블이 매칭 되어야함
#       return x, y
#
# def sequential_batch(batch_size):
#       모델 테스트용
#       return x, file_name   csv 만들어야 되니까

from PIL import Image
import numpy as np
import csv
import random
import tensorflow as tf


class DataSet:
    def __init__(self, imagePath, csvPath):
        f = open(csvPath, 'r', encoding="utf-8")
        reader = list(csv.reader(f))
        reader.pop(0)
        f.close()
        self.numData = len(reader)

        self.imageData = np.zeros([self.numData, 128, 128, 3], dtype=np.float32)
        self.imageLable = np.zeros([self.numData], dtype=np.uint8)
        self.fileName = []
        self.currentIdx = 0

        for i, line in enumerate(reader):
            tempImg = Image.open(imagePath + line[0], "r").convert('RGB')
            # self.imageData[i, :] = np.array(tempImg)
            self.imageData[i, :] = np.array(tempImg) / 255.0
            self.imageLable[i] = np.array(line[1])
            self.fileName.append(line[0])

    def random_batch(self, batch_size):
        idx = random.sample(list(range(0, self.numData)), batch_size)
        x = self.imageData[idx, :]
        y = self.imageLable[idx]
        return x, y

    def sequential_batch(self, batch_size):
        if self.currentIdx + batch_size <= self.numData:
            x = self.imageData[self.currentIdx:self.currentIdx + batch_size]
            n = self.fileName[self.currentIdx:self.currentIdx + batch_size]
            self.currentIdx += batch_size
            return x, n, False
        else:
            x = self.imageData[self.currentIdx:]
            n = self.fileName[self.currentIdx:]
            return x, n, True


def print_write(s, file, mode=None):
    if isinstance(file, str):
        if mode is None:
            mode = 'a'
        f = open(file, mode)
        print(s, end='')
        f.write(s)
        f.close()
    else:
        print(s, end='')
        file.write(s)


def get_tf_config():
    # config = tf.ConfigProto(log_device_placement=True)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return config


def exclude_batch_norm(name):
    return 'batch_normalization' not in name


def csv_save(file, pred):
    f = open(file, 'w')
    f.write('filename,prediction\n')
    for n, p in pred:
        f.write('%s,%s\n' % (n, p))
    f.close()
    return


def main():
    imagePath = "../_Data/1. 얼굴사진분류 데이터/image_v2/face_images_128x128/"
    csvPath = "../_Data/1. 얼굴사진분류 데이터/ml_8_faceclassifier_train.csv"
    datee = DataSet(imagePath, csvPath)


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

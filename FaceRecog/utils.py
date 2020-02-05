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
            if len(line) > 1:
                self.imageLable[i] = np.array(line[1])
            self.fileName.append(line[0])

    def sequential_idx_reset(self):
        self.currentIdx = 0
        return

    def train_batch(self, batch_size):
        idx = random.sample(list(range(0, self.numData)), batch_size)
        x = self.imageData[idx, :]
        y = self.imageLable[idx]
        return x, y

    def val_batch(self, batch_size):
        if self.currentIdx + batch_size <= self.numData:
            x = self.imageData[self.currentIdx:self.currentIdx + batch_size]
            y = self.imageLable[self.currentIdx:self.currentIdx + batch_size]
            self.currentIdx += batch_size
            return x, y, False
        else:
            x = self.imageData[self.currentIdx:]
            y = self.imageLable[self.currentIdx:]
            return x, y, True

    def infer_batch(self, batch_size):
        if self.currentIdx + batch_size <= self.numData:
            x = self.imageData[self.currentIdx:self.currentIdx + batch_size]
            n = self.fileName[self.currentIdx:self.currentIdx + batch_size]
            self.currentIdx += batch_size
            return x, n, False
        else:
            x = self.imageData[self.currentIdx:]
            n = self.fileName[self.currentIdx:]
            return x, n, True


def csv_split(csvfile, ratio=0.1):
    f = open(csvfile, 'r', encoding="utf-8")
    reader = list(csv.reader(f))
    f.close()

    train_f = open(csvfile.split('.')[0] + '_split.csv', 'w')
    val_f = open(csvfile.split('.')[0] + '_val.csv', 'w')
    head = reader.pop(0)
    train_f.write('%s\n' % ','.join(head))
    val_f.write('%s\n' % ','.join(head))

    for line in reader:
        if random.random() < ratio:
            val_f.write('%s\n' % ','.join(line))
        else:
            train_f.write('%s\n' % ','.join(line))

    train_f.close()
    val_f.close()
    return


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


def flip(x: tf.Tensor) -> tf.Tensor:
    """Flip augmentation

    Args:
        x: Image to flip

    Returns:
        Augmented image
    """
    x = tf.image.random_flip_left_right(x)

    return x


def color(x: tf.Tensor) -> tf.Tensor:
    """Color augmentation

    Args:
        x: Image

    Returns:
        Augmented image
    """
    x = tf.image.random_hue(x, 0.08)
    x = tf.image.random_saturation(x, 0.6, 1.6)
    x = tf.image.random_brightness(x, 0.05)
    x = tf.image.random_contrast(x, 0.7, 1.3)
    return x


def images_augment(x):
    x = tf.cond(tf.random_uniform([], 0, 1) > 0.75, lambda: flip(x), lambda: x)
    x = tf.cond(tf.random_uniform([], 0, 1) > 0.75, lambda: color(x), lambda: x)
    return tf.clip_by_value(x, 0, 1)


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

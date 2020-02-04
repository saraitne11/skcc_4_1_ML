<<<<<<< Updated upstream
######################## BEGIN LICENSE BLOCK ########################
# The Original Code is Mozilla Universal charset detector code.
#
# The Initial Developer of the Original Code is
#          Simon Montagu
# Portions created by the Initial Developer are Copyright (C) 2005
# the Initial Developer. All Rights Reserved.
#
# Contributor(s):
#   Mark Pilgrim - port to Python
#   Shy Shalom - original C code
#   Shoshannah Forbes - original C code (?)
#
# This library is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License as published by the Free Software Foundation; either
# version 2.1 of the License, or (at your option) any later version.
=======
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
>>>>>>> Stashed changes
#
# This library is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public
# License along with this library; if not, write to the Free Software
# Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA
# 02110-1301  USA
######################### END LICENSE BLOCK #########################

# 255: Control characters that usually does not exist in any text
# 254: Carriage/Return
# 253: symbol (punctuation) that does not belong to word
# 252: 0 - 9

# Windows-1255 language model
# Character Mapping Table:
WIN1255_CHAR_TO_ORDER_MAP = (
255,255,255,255,255,255,255,255,255,255,254,255,255,254,255,255,  # 00
255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,  # 10
253,253,253,253,253,253,253,253,253,253,253,253,253,253,253,253,  # 20
252,252,252,252,252,252,252,252,252,252,253,253,253,253,253,253,  # 30
253, 69, 91, 79, 80, 92, 89, 97, 90, 68,111,112, 82, 73, 95, 85,  # 40
 78,121, 86, 71, 67,102,107, 84,114,103,115,253,253,253,253,253,  # 50
253, 50, 74, 60, 61, 42, 76, 70, 64, 53,105, 93, 56, 65, 54, 49,  # 60
 66,110, 51, 43, 44, 63, 81, 77, 98, 75,108,253,253,253,253,253,  # 70
124,202,203,204,205, 40, 58,206,207,208,209,210,211,212,213,214,
215, 83, 52, 47, 46, 72, 32, 94,216,113,217,109,218,219,220,221,
 34,116,222,118,100,223,224,117,119,104,125,225,226, 87, 99,227,
106,122,123,228, 55,229,230,101,231,232,120,233, 48, 39, 57,234,
 30, 59, 41, 88, 33, 37, 36, 31, 29, 35,235, 62, 28,236,126,237,
238, 38, 45,239,240,241,242,243,127,244,245,246,247,248,249,250,
  9,  8, 20, 16,  3,  2, 24, 14, 22,  1, 25, 15,  4, 11,  6, 23,
 12, 19, 13, 26, 18, 27, 21, 17,  7, 10,  5,251,252,128, 96,253,
)

# Model Table:
# total sequences: 100%
# first 512 sequences: 98.4004%
# first 1024 sequences: 1.5981%
# rest  sequences:      0.087%
# negative sequences:   0.0015%
HEBREW_LANG_MODEL = (
0,3,3,3,3,3,3,3,3,3,3,2,3,3,3,3,3,3,3,3,3,3,3,2,3,2,1,2,0,1,0,0,
3,0,3,1,0,0,1,3,2,0,1,1,2,0,2,2,2,1,1,1,1,2,1,1,1,2,0,0,2,2,0,1,
3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,2,2,2,2,
1,2,1,2,1,2,0,0,2,0,0,0,0,0,1,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,
3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,2,2,2,
1,2,1,3,1,1,0,0,2,0,0,0,1,0,1,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,
3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,1,0,1,2,2,1,3,
1,2,1,1,2,2,0,0,2,2,0,0,0,0,1,0,1,0,0,0,1,0,0,0,0,0,0,1,0,1,1,0,
3,3,3,3,3,3,3,3,3,3,3,3
)
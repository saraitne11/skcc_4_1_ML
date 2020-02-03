import csv
import random
import numpy as np

filePath = '../_data/ml_6_spacing_train.csv'

ALPHABET = {c: i for i, c in enumerate('Pabcdefghijklmnopqrstuvwxyz')}
PAD = 'P'


def word2idx(word, max_len, dtype=np.uint8):
    word = word + PAD * (max_len-len(word))
    return np.array(list(map(lambda c: ALPHABET[c], word)), dtype=dtype)


def readData(fileName):
    f = open(fileName, 'r', encoding='utf-8')
    lines = list(csv.reader(f))
    f.close()

    max_len = 0
    compounds = [i[0] for i in lines[1:]]
    for word in compounds:
        if (max_len < len(word)):
            max_len = len(word)

    flag = True
    if (len(lines[0]) == 1):
        flag = False

    alpha_indices = np.ones((len(compounds), max_len), dtype=np.uint8)
    split_indices = np.zeros((len(compounds), max_len), dtype=np.uint8)
    word_compound = []

    index = 0
    for line in lines[1:]:
        word_compound.append([line[0], []])
        alpha_indices[index] = word2idx(line[0], max_len)
        if (flag):
            inner_index = 0
            for data in line[1:]:
                if (len(data) == 0):
                    break
                else:
                    split_indices[index][inner_index] = 1
                    word_compound[index][1].append(data)
                    inner_index += len(data)
        index += 1
    return alpha_indices, split_indices, word_compound


class DataSet:
    def __init__(self):
        self.train_path = '../_data/ml_6_spacing_train.csv'
        self.test_path = '../_data/ml_6_spacing_test.csv'

        self.x, self.y, self.compound = readData(self.train_path)
        self.seq_len = [len(i[0]) for i in self.compound]
        self.num_data = len(self.x)

        self.test_x, self.test_y, self.test_compound = readData(self.test_path)
        self.test_seq_len = [len(i[0]) for i in self.test_compound]
        self.test_num_data = len(self.test_x)

        self.sequential_index = 0

    def random_batch(self, batch_size):
        choose = random.sample(list(range(0, self.num_data)), batch_size)
        # x = np.array(self.x)[choose]
        # y = np.array(self.y)[choose]
        x = self.x[choose]
        y = self.y[choose]

        return x, y

    def sequentital_batch(self, batch_size):
        x = self.test_x[self.sequential_index:][:batch_size]
        self.sequential_index += batch_size

        if (self.sequential_index >= self.test_num_data):
            return False, x

        return True, x


if __name__ == '__main__':
    dataSet = DataSet()

# TODO
# ReadData 클래스로 만들기
# example
# class DataSet
# def __init__(csv 파일 경로):
#       self.x = [데이터 개수, 가장 긴 복합어의 글자 개수]
#                   [[5, 2, 3, 4, 2, 3, 0, 0, 0, 0], [6, 5, 4, 2, 1, 3, 0, 0, 0, 0], ...]
#       self.y = [데이터 개수, 가장 긴 복합어의 글자 개수]
#                   [[1, 0, 0, 0, 1, 0, 0, 0, 0, 0], [1, 0, 0, 1, 0, 0, 0, 0, 0, 0], ...]
#       self.seq_len = []
#       self.compounds = ['hamburger', ['ham', 'burger']]
#       self.num_data = 데이터 개수
#       self.sequential_index = 0
#
# def random_batch(batch_size):
#       모델 학습용
#       x = 랜덤한 batch_size개의 데이터    [batch_size, 128, 128, 3]
#       y = 랜덤한 batch_size개의          [batch_size]
#           이미지 데이터와 레이블이 매칭 되어야함
#       return x, y
#
# def sequential_batch(batch_size):
#       모델 테스트용
#       return x

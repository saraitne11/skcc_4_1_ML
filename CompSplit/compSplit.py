import csv
import numpy as np

filePath = '../_Data/ml_6_spacing_train.csv'

def readData(fileName):
    f = open(fileName, 'r', encoding='utf-8')
    lines = list(csv.reader(f))
    lines.pop(0)
    f.close()

    retData = []

    for line in lines:
        row = [line[0], [], np.zeros(len(line[0]), dtype=np.uint8)]
        line.pop(0)
        index = 0
        for data in line:
            if (len(data) == 0):
                break
            else:
                row[2][index] = 1
                index += len(data)
                row[1].append(data)
        retData.append(row)

    return retData


if __name__ == '__main__':
    print(readData(filePath))


# TODO
# ReadData 클래스로 만들기
# example
# class DataSet
# def __init__(csv 파일 경로):
#       self.x = [데이터 개수, 가장 긴 복합어의 글자 개수]
#                   [[5, 2, 3, 4, 2, 3, 0, 0, 0, 0], [6, 5, 4, 2, 1, 3, 0, 0, 0, 0], ...]
#       self.y = [데이터 개수, 가장 긴 복합어의 글자 개수]
#                   [[1, 0, 0, 0, 1, 0, 0, 0, 0, 0], [1, 0, 0, 1, 0, 0, 0, 0, 0, 0], ...]
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
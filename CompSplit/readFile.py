import csv
import numpy as np

filePath = '../_data/ml_6_spacing_train.csv'

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
from matplotlib import pyplot as plt
import numpy as np
import csv
import pandas as pd
from pandas import Series, DataFrame
from itertools import groupby


def main():
    csvPath = './3. 환자정보 데이터/ml_7_medicalalert_train_v1.csv'
    f = open(csvPath, 'r', encoding="utf-8")
    reader = list(csv.reader(f))
    col_name = reader.pop(0)
    reader = sorted(reader, key=lambda x: (x[0], x[1]))

    data = groupby(reader, lambda x: x[0])

    for key, items in data:
        print(key, len(list(items)))
        for item in items:
            print(item)




if __name__ == '__main__':
    main()

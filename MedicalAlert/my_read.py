import sys
from datetime import datetime
import numpy as np


PATIENTID = 0
TIMESTAMP = 1
GENDER = 2
AGE = 3
DI = 4
COPD = 5
CHF = 6
HT = 7
AFID = 8
W = 9
PW = 10
BPS = 11
BPD = 12
SPO2 = 13
HR = 14
GL = 15
ALERT = 16


VECTOR_LEN = 17
PARAM_VEC_LEN = 14

UPPER_BOUND = 80
LOWER_BOUND = 20

def get_threshold(x):
    a = np.array(x)
    upper_quartile = np.nanpercentile(a, UPPER_BOUND, axis=1)
    lower_quartile = np.nanpercentile(a, LOWER_BOUND, axis=1)

    return np.concatenate((upper_quartile[:, np.newaxis], lower_quartile[:, np.newaxis]), axis=1)


def print_2d_array(arr):
    row, col = np.shape(arr)
    for i in range(row):
        for j in range(col):
            print('%0.3f' % arr[i, j], end=' ')
        print()
    return


def none_or_float(x):
    if not x:           # x == ''
        return float('nan')
    else:
        return float(x)


def line_preproc(line):
    if len(line) != VECTOR_LEN:
        print('len(line) %d is not 17' % len(line))
        sys.exit(-1)

    if not line[PATIENTID] or line[PATIENTID] == 'patientid':
        return None

    if not line[TIMESTAMP]:
        print('line TimeStamp Error %d' % len(line))
    else:
        line[TIMESTAMP] = datetime.strptime(line[TIMESTAMP], '%Y-%m-%d %H:%M:%S')

    if line[GENDER] == 'M':
        line[GENDER] = 0.0
    elif line[GENDER] == 'F':
        line[GENDER] = 1.0
    elif not line[GENDER]:      # if line[GENDER] == ''
        line[GENDER] = None
    else:
        print('Gender Error', line)
        sys.exit(-1)

    line[AGE] = none_or_float(line[AGE])        # 나이
    line[DI] = none_or_float(line[DI])          # 1이면 당뇨병이 있음을 나타낸다
    line[COPD] = none_or_float(line[COPD])      # 1이면 만성 폐질환이 있음을 나타낸다
    line[CHF] = none_or_float(line[CHF])        # 1이면 울혈성 심부전증이 있음을 나타낸다
    line[HT] = none_or_float(line[HT])          # 1이면 고혈압이 있음을 나타낸다
    line[AFID] = none_or_float(line[AFID])      # 1이면 심방세동이 있음을 나타낸다
    line[W] = none_or_float(line[W])            # 측정된 몸무게가 Kg으로 들어온다
    line[PW] = none_or_float(line[PW])          # 가장 최근에 측정된 몸무게가 Kg으로 들어온다.
    line[BPS] = none_or_float(line[BPS])        # 수축기 혈압
    line[BPD] = none_or_float(line[BPD])        # 이완기 혈압
    line[SPO2] = none_or_float(line[SPO2])      # 혈중 산소 포화도를 나타낸다
    line[HR] = none_or_float(line[HR])          # 맥박을 나타낸다
    line[GL] = none_or_float(line[GL])          # 혈당을 나타낸다

    if line[ALERT] == 'Yes':
        line[ALERT] = 1.0
    elif line[ALERT] == 'No':
        line[ALERT] = 0.0
    else:
        print('Alert Error', line)
        sys.exit(-1)
    return line


class TrainData:
    def __init__(self, train_csv):
        f = open(train_csv, 'r', encoding='utf-8')
        lines = list(map(lambda x: x.rstrip().split(','), f.readlines()))
        f.close()

        self.head = lines[0]
        # print(self.head)
        # print(len(self.head))

        self.lines = []
        for line in lines[1:]:
            preproc = line_preproc(line)
            if preproc:
                self.lines.append(preproc)

        self.num_lines = len(self.lines)

        self.patients = list(set(map(lambda x: x[PATIENTID], self.lines)))
        self.num_patients = len(self.patients)

        self.group_by_patient = {p: [] for p in self.patients}
        for line in self.lines:
            self.group_by_patient[line[PATIENTID]].append(line)

        self.max_seq_len = 0
        for patient in self.patients:
            self.group_by_patient[patient].sort(key=lambda x: x[TIMESTAMP])
            if len(self.group_by_patient[patient]) > self.max_seq_len:
                self.max_seq_len = len(self.group_by_patient[patient])
        # print(self.max_seq_len)


        from pprint import pprint
        # pprint(self.lines[:300], width=300)
        # print(self.num_lines)
        # pprint(self.group_by_patient, width=300)
        # pprint(self.group_by_patient['318614146924878026'], width=300)

        self.data_init()
        # pprint(self.group_by_patient, width=300)
        return

    def get_patient_means(self):
        means = {}
        for patient in self.patients:
            temp_data = self.group_by_patient[patient]
            temp_data = np.array(list(map(lambda x: x[GENDER:ALERT], temp_data)))
            temp_mean = np.nanmean(temp_data, axis=0)
            temp_mean[np.isnan(temp_mean)] = 0.0
            means[patient] = temp_mean
        return means


    def normalization(self):
        # None and outlier 처리
        # 환자의 평균값
        # 0~1 사이의 값으로 변환
        return

    def data_init(self):
        self.data = np.zeros([self.num_patients, self.max_seq_len, PARAM_VEC_LEN])

        outlier_mask = [[], [], [], [], [], [], []]
        for values in self.group_by_patient.values():
            for index in range(W, ALERT):
                outlier_mask[index - W] = outlier_mask[index - W] + list(list(zip(*values))[index])

        # outlier의 경계값 찾기
        self.threshold = get_threshold(outlier_mask)

        # outlier NaN으로 바꾸기
        for patient in self.patients:
            for idx in range(len(self.group_by_patient[patient])):
                temp = self.group_by_patient[patient][idx][W:ALERT]
                for index in range(len(temp)):
                    if temp[index] == np.nan or temp[index] < self.threshold[index][1] or temp[index] > self.threshold[index][0]:
                        temp[index] = np.nan
                self.group_by_patient[patient][idx][W:ALERT] = temp

        self.means_by_patient = self.get_patient_means()

        # for item in self.means_by_patient.values():
            # print(np.shape(item))

        # NaN mean 데이터로 바꾸기
        for patient in self.patients:
            for idx in range(len(self.group_by_patient[patient])):
                for j in range(GENDER, ALERT):
                    if np.isnan(self.group_by_patient[patient][idx][j]):
                        self.group_by_patient[patient][idx][j] = self.means_by_patient[patient][j - GENDER]

        # patient id 별로 self.data 채워 넣기

        # outlier_removed_data = []
        # for values in outlier_mask:
        #     outlier_removed_data.append(removeOutliers(values))
        # outlier_removed_data = np.array(outlier_removed_data)

        # odata_index = 0
        # index = 0
        # for values in self.group_by_patient.values():
        #     inner_index = 0
        #     for i in range(len(values)):
        #         self.group_by_patient
        #     for value in values:
        #         self.data[index, inner_index] = np.array(value[2:16])
        #         self.data[index, inner_index, 7:14] = np.array(outlier_removed_data[:, odata_index])
        #         inner_index += 1
        #         odata_index += 1
        #     index += 1


        # numpy array로 만들기
        # [num_patient(185), max_seq_len(402), 14]
        # self.data = np.zeros([num_patient(185), max_seq_len(402), 14])
        self.normalization()
        return

    def train_batch(self):
        return


class TestData:
    def __init__(self):
        return

    def test_batch(self, train_data: TrainData):
        return



if __name__ == '__main__':
    data = TrainData('../_Data/ml_10_medicalalert_train.csv')

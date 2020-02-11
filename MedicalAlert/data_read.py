import sys
from datetime import datetime
import numpy as np
import random
import bisect
from pprint import pprint


PATIENTID = 0
TIMESTAMP = 1
GENDER = 2
AGE = 3
DI = 4
COPD = 5
CHF = 6
HT = 7
AFIB = 8
W = 9
PW = 10
BPS = 11
BPD = 12
SPO2 = 13
HR = 14
GL = 15
ALERT = 16

YES = 2.0
NO = 1.0

LINE_LEN = 17
NUM_FEATURES = 14


def print_2d_array(arr):
    row, col = np.shape(arr)
    for i in range(row):
        for j in range(col):
            print('%5.1f' % arr[i, j], end=' ')
        print()
    return


def print_1d_array(arr):
    col = np.shape(arr)[0]
    for i in range(col):
        print('%5.1f' % arr[i], end=' ')
    print()
    return


def count_nan(arr):
    k = np.isnan(arr)
    return np.sum(k.astype(np.int8))


def none_or_float(x):
    if not x:           # x == ''
        return np.nan
    else:
        return float(x)


def line_preproc(line, is_train=True):
    if len(line) != LINE_LEN:
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
    line[AFIB] = none_or_float(line[AFIB])      # 1이면 심방세동이 있음을 나타낸다
    line[W] = none_or_float(line[W])            # 측정된 몸무게가 Kg으로 들어온다
    line[PW] = none_or_float(line[PW])          # 가장 최근에 측정된 몸무게가 Kg으로 들어온다.
    line[BPS] = none_or_float(line[BPS])        # 수축기 혈압
    line[BPD] = none_or_float(line[BPD])        # 이완기 혈압
    line[SPO2] = none_or_float(line[SPO2])      # 혈중 산소 포화도를 나타낸다
    line[HR] = none_or_float(line[HR])          # 맥박을 나타낸다
    line[GL] = none_or_float(line[GL])          # 혈당을 나타낸다

    if is_train:
        if line[ALERT] == 'Yes':
            line[ALERT] = YES
        elif line[ALERT] == 'No':
            line[ALERT] = NO
        else:
            print('Alert Error', line)
            sys.exit(-1)
    return line


class TrainData:
    def __init__(self, train_csv,
                 yes_weight=200.0, no_weight=1.0,
                 upper_percet=90, lower_percent=20):
        self.yes_weight = yes_weight
        self.no_weight = no_weight
        self.upper_percent = upper_percet
        self.lower_percent = lower_percent
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
        # pprint(self.lines[:300], width=300)
        # print(self.num_lines)

        self.patients = list(set(map(lambda x: x[PATIENTID], self.lines)))
        self.num_patients = len(self.patients)

        self.group_by_patient = {p: [] for p in self.patients}
        for line in self.lines:
            self.group_by_patient[line[PATIENTID]].append(line)

        self.max_seq_len = 0
        self.num_data = 0
        for patient in self.patients:
            self.group_by_patient[patient].sort(key=lambda x: x[TIMESTAMP])
            num_data_per_patient = len(self.group_by_patient[patient])
            self.num_data += num_data_per_patient
            if num_data_per_patient > self.max_seq_len:
                self.max_seq_len = len(self.group_by_patient[patient])
        # pprint(self.group_by_patient, width=300)
        # print(self.num_data)
        # print(self.max_seq_len)

        # =================================================
        # self.upper, self.lower = self.get_boundary()
        # =================================================
        # print_1d_array(self.upper)
        # print_1d_array(self.lower)
        # print(np.shape(self.upper), np.shape(self.lower))

        # =================================================
        # self.outlier2nan(self.upper, self.lower)
        # =================================================
        # pprint(self.group_by_patient, width=300)

        # 각 환자별, 그리고 피쳐별[W~GL]로 평균
        # 185[환자 수] by 7[피쳐 수]
        # =================================================
        # self.patient_means = self.get_patient_means()
        # =================================================
        # for patient in self.patients:
        #     print_1d_array(self.patient_means[patient])
        # print(len(self.patient_means))

        # self.nan2mean(self.patient_means)
        self.nan2minus1()
        # pprint(self.group_by_patient, width=300)

        # self.max_by_features, self.min_by_features = self.get_min_max_by_features()
        # print_1d_array(self.max_by_features)
        # print_1d_array(self.min_by_features)

        # self.min_max_normalization(self.max_by_features, self.min_by_features)
        # pprint(self.group_by_patient, width=300)

        self.check_nan()

        self.x, self.y, self.weight, self.seq_len = self.data_init()
        # print_2d_array(self.x[121])
        # print_1d_array(self.y[121])
        # print(np.shape(self.x))
        # print(np.shape(self.y))
        # print(np.shape(self.seq_len))
        # print(count_nan(self.x))
        # print(count_nan(self.y))
        # print(count_nan(self.seq_len))
        return

    def data_init(self):
        target_features = [GENDER, AGE, DI, COPD, CHF, HT, AFIB, W, PW, BPS, BPD, SPO2, HR, GL]
        x = np.zeros([self.num_patients, self.max_seq_len, len(target_features)], dtype=np.float32)
        y = np.zeros([self.num_patients, self.max_seq_len], dtype=np.int32)
        weights = np.zeros([self.num_patients, self.max_seq_len], dtype=np.float32)
        seq_len = np.zeros([self.num_patients], dtype=np.float32)
        for patient_idx, patient in enumerate(self.patients):
            num_data_per_patient = len(self.group_by_patient[patient])
            seq_len[patient_idx] = num_data_per_patient
            for time_step in range(num_data_per_patient):
                x[patient_idx, time_step, :] = self.group_by_patient[patient][time_step][GENDER:ALERT]

                alert = self.group_by_patient[patient][time_step][ALERT]
                y[patient_idx, time_step] = alert

                if alert >= YES:
                    weights[patient_idx, time_step] = self.yes_weight
                else:
                    weights[patient_idx, time_step] = self.no_weight

        return x, y, weights, seq_len

    def min_max_normalization(self, _max, _min):
        target_features = [AGE, W, PW, BPS, BPD, SPO2, HR, GL]
        for patient in self.patients:
            num_data_per_patient = len(self.group_by_patient[patient])
            for i in range(num_data_per_patient):
                for k, j in enumerate(target_features):
                    value = self.group_by_patient[patient][i][j]
                    normalized_value = (value-_min[k]) / (_max[k]-_min[k])
                    self.group_by_patient[patient][i][j] = normalized_value
        return

    def get_min_max_by_features(self):
        # [24997, 8]
        # 모든 환자들의 features
        # features : 나이 몸무게 최근몸무게 수축혈압 이완혈압 산소포화도 맥박 혈당
        target_features = [AGE, W, PW, BPS, BPD, SPO2, HR, GL]
        w2gl = np.zeros([self.num_data, len(target_features)], dtype=np.float32)
        idx = 0
        for patient in self.patients:
            for features in self.group_by_patient[patient]:
                w2gl[idx, :] = [features[AGE]] + features[W:ALERT]
                idx += 1
        # print_2d_array(w2gl)
        # print(np.shape(w2gl))
        _max = np.max(w2gl, axis=0)
        _min = np.min(w2gl, axis=0)
        return _max, _min

    def check_nan(self):
        flag = False
        for patient in self.patients:
            num_data_per_patient = len(self.group_by_patient[patient])
            for i in range(num_data_per_patient):
                temp = self.group_by_patient[patient][i]
                for x in temp[GENDER:]:
                    if np.isnan(x):
                        print(patient, temp)
                        flag = True
                        continue
        if flag:
            sys.exit(-1)
        return

    def nan2mean(self, patient_means):
        for patient in self.patients:
            num_data_per_patient = len(self.group_by_patient[patient])
            for i in range(num_data_per_patient):
                temp = self.group_by_patient[patient][i]
                # 지병 여부가 nan 이면 0.0 으로 변환
                for j in range(DI, W):
                    if np.isnan(temp[j]):
                        temp[j] = 0.0
                # 수치가 nan 이면 평균값으로 변환
                for j in range(W, ALERT):
                    if np.isnan(temp[j]):
                        temp[j] = patient_means[patient][j-W]

                self.group_by_patient[patient][i] = temp
        return

    def nan2minus1(self):
        for patient in self.patients:
            num_data_per_patient = len(self.group_by_patient[patient])
            for i in range(num_data_per_patient):
                temp = self.group_by_patient[patient][i]
                # nan 이면 -1.0 으로 변환
                for j in range(DI, ALERT):
                    if np.isnan(temp[j]):
                        temp[j] = -1.0
                self.group_by_patient[patient][i] = temp
        return

    def get_patient_means(self):
        means = {}
        for patient in self.patients:
            patient_data = np.array(list(map(lambda x: x[W:ALERT], self.group_by_patient[patient])))
            patient_mean = np.nanmean(patient_data, axis=0)
            patient_mean[np.isnan(patient_mean)] = 0.0
            means[patient] = patient_mean
        return means

    def get_boundary(self):
        # [24997, 7]
        # 모든 환자들의 몸무게부터 혈당까지의 features
        target_features = [W, PW, BPS, BPD, SPO2, HR, GL]
        w2gl = np.zeros([self.num_data, len(target_features)], dtype=np.float32)
        idx = 0
        for patient in self.patients:
            for features in self.group_by_patient[patient]:
                w2gl[idx, :] = features[W:ALERT]
                idx += 1
        # print_2d_array(w2gl)
        # print(np.shape(w2gl))
        upper = np.nanpercentile(w2gl, self.upper_percent, axis=0)
        lower = np.nanpercentile(w2gl, self.lower_percent, axis=0)
        return upper, lower

    def outlier2nan(self, upper, lower):
        for patient in self.patients:
            num_data_per_patient = len(self.group_by_patient[patient])
            for i in range(num_data_per_patient):
                temp = self.group_by_patient[patient][i][W:ALERT]
                for j in range(ALERT-W):
                    if temp[j] == np.nan or temp[j] > upper[j] or temp[j] < lower[j]:
                        temp[j] = np.nan
                self.group_by_patient[patient][i][W:ALERT] = temp

    def train_batch(self, batch_size):
        choose = random.sample(list(range(0, self.num_patients)), batch_size)
        x = self.x[choose]
        y = self.y[choose]
        weight = self.weight[choose]
        seq_len = self.seq_len[choose]
        return x, y, weight, seq_len


class TestData:
    def __init__(self, test_csv, train_data: TrainData):
        f = open(test_csv, 'r', encoding='utf-8')
        lines = list(map(lambda x: x.rstrip().split(','), f.readlines()))
        f.close()

        self.train_data = train_data

        self.head = lines[0]
        self.lines = []
        for line in lines[1:]:
            line.append(line.pop(0))
            preproc = line_preproc(line, False)
            if preproc:
                self.lines.append(preproc)

        self.num_lines = len(self.lines)

        self.patients = list(set(map(lambda x: x[PATIENTID], self.lines)))
        self.num_patients = len(self.patients)

        self.sequential_index = 0

        # self.upper, self.lower = train_data.upper, train_data.lower

        # self.outlier2nan(self.upper, self.lower)

        # self.patient_means = train_data.patient_means

        # self.nan2mean(self.patient_means)
        self.nan2minus1()

        # self.max_by_features, self.min_by_features = train_data.max_by_features, train_data.min_by_features

        # self.min_max_normalization(self.max_by_features, self.min_by_features)

        self.check_nan()
        return

    def min_max_normalization(self, _max, _min):
        target_features = [AGE, W, PW, BPS, BPD, SPO2, HR, GL]
        for i in range(self.num_lines):
            for k, j in enumerate(target_features):
                value = self.lines[i][j]
                normalized_value = (value-_min[k]) / (_max[k]-_min[k])
                self.lines[i][j] = normalized_value
        return

    def check_nan(self):
        flag = False
        for i in range(self.num_lines):
            temp = self.lines[i]
            for x in temp[GENDER:ALERT]:
                if np.isnan(x):
                    print(self.lines[i][PATIENTID], temp)
                    flag = True
                    continue
        if flag:
            sys.exit(-1)
        return

    def nan2mean(self, patient_means):
        for i in range(self.num_lines):
            temp = self.lines[i]
            patient = temp[PATIENTID]

            # 지병 여부가 nan 이면 0.0 으로 변환
            for j in range(DI, W):
                if np.isnan(temp[j]):
                    temp[j] = 0.0

            # 수치가 nan 이면 평균값으로 변환
            for j in range(W, ALERT):
                if np.isnan(temp[j]):
                    temp[j] = patient_means[patient][j - W]

            self.lines[i] = temp
        return

    def nan2minus1(self):
        for i in range(self.num_lines):
            temp = self.lines[i]

            # nan 이면 -1.0 으로 변환
            for j in range(DI, ALERT):
                if np.isnan(temp[j]):
                    temp[j] = -1.0

            self.lines[i] = temp
        return

    def outlier2nan(self, upper, lower):
        for i in range(self.num_lines):
            temp = self.lines[i][W:ALERT]
            for j in range(ALERT-W):
                if temp[j] == np.nan or temp[j] > upper[j] or temp[j] < lower[j]:
                    temp[j] = np.nan
            self.lines[i][W:ALERT] = temp

    def test_batch(self):
        cur_data = self.lines[self.sequential_index]
        self.sequential_index += 1

        flag = False
        if self.sequential_index == self.num_lines:
            flag = True

        patient_id = cur_data[PATIENTID]
        cur_time = cur_data[TIMESTAMP]

        time_stamps = [i[1] for i in self.train_data.group_by_patient[patient_id]]
        crop_index = bisect.bisect_left(time_stamps, cur_time)

        if crop_index == 0:
            return (np.array(cur_data[GENDER:ALERT]))[np.newaxis, :], 1, flag

        index = self.train_data.patients.index(patient_id)

        x = self.train_data.x[index, :crop_index]
        temp = np.array(cur_data[GENDER:ALERT])
        temp = temp[np.newaxis, :]
        x = np.concatenate([x, temp])
        return x, np.shape(x)[0], flag


if __name__ == '__main__':
    data = TrainData('../_Data/ml_10_medicalalert_train.csv')
    t_data = TestData('../_Data/ml_10_medicalalert_test.csv', data)

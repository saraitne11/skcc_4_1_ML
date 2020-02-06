import sys
from datetime import datetime


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


def none_or_float(x):
    if not x:           # x == ''
        return None
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
        print(self.max_seq_len)

        # from pprint import pprint
        # pprint(self.lines[:300], width=300)
        # print(self.num_lines)
        # pprint(self.group_by_patient, width=300)
        return


if __name__ == '__main__':
    data = TrainData('../_Data/ml_10_medicalalert_train.csv')

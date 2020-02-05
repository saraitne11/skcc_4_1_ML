import numpy as np
import pandas as pd
import random
import math
import bisect

outlier_value = ['w', 'pw', 'bps', 'bpd', 'spo2', 'hr', 'gl']
all_value = ['age', 'di', 'copd', 'chf', 'ht', 'afib'] + outlier_value


def remove_outlier(df):
   for col in outlier_value:
       low = df[col].quantile(.05)
       high = df[col].quantile(.95)
       for i in df.index:
           val = df.at[i, col]
           if math.isnan(val) or val < low or val > high or val == 0:
               df.at[i, col] = np.nan


def fillNa(df):
   for col in outlier_value:
       df[col] = df.groupby('patientid')[col].apply(lambda x: x.fillna(x.mean()))
       df[col] = df[col].fillna(df[col].mean())


def normalization(df):
    for col in all_value:
        df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())


def RepresentsInt(s):
   try:
       int(s)
       return True
   except ValueError:
       return False


def AlterGender(s):
    if s == 'M':
        return 1
    else:
        return 0


def AlertToValue(s):
   if s == 'Yes':
       return 1
   else:
       return 0


def trainReadData(fileName):
   data_ = pd.read_csv(fileName, encoding='utf-8')
   data_['gender'] = data_['gender'].map(lambda x : AlterGender(x))
   data_ = data_.sort_values(by=['patientid','timestamp'])
   remove_outlier(data_)
   fillNa(data_)
   normalization(data_)
   reader = data_.values.tolist()

   patient_id = []
   for each_row in reader:
       patient_id.append(each_row[0])
   patient_id = list(set(patient_id))
   num_data = len(patient_id)

   determinants = {}
   time_table = {}
   alerts = {}
#  gender = {}
   for pid in patient_id:
       determinants[pid] = []
       alerts[pid] = []
       time_table[pid] = []

   for each_row in reader:
       determinants[each_row[0]].append(each_row[2:16])
       time_table[each_row[0]].append(each_row[1])
       alerts[each_row[0]].append(AlertToValue(each_row[16]))

   max_len = 0
   for pid in patient_id:
       max_len = max(max_len, len(determinants[pid]))

   x = np.zeros([num_data, max_len, 14], dtype=np.float32)
   y = np.zeros([num_data, max_len], dtype=np.uint8)
   time_stamp = np.zeros([num_data, max_len], dtype=object)
   index = 0
   for pid in patient_id:
       inner_index = 0
       for deter in determinants[pid]:
           x[index][inner_index] = np.array(deter)
           inner_index += 1

       inner_index = 0
       for alert in alerts[pid]:
           y[index][inner_index] = alert
           inner_index += 1

       inner_index = 0
       for cur_time in time_table[pid]:
           time_stamp[index][inner_index] = cur_time
           inner_index += 1
       index += 1

   return x, y, patient_id, time_stamp

def testReadData(fileName):
   data_ = pd.read_csv(fileName, encoding='utf-8')
   data_['gender'] = data_['gender'].map(lambda x : AlterGender(x))


class DataSet:
   def __init__(self, train_path, test_path):
       self.train_path = train_path
       self.test_path = test_path

       self.train_x, self.train_y, self.train_patient_id, self.train_time_stamp = trainReadData(self.train_path)
       self.train_num_data = len(self.train_x)

       self.test_x, self.test_y, self.test_patient_id, self.test_time_stamp = testReadData(self.test_path)
       self.test_num_data = len(self.test_x)

       self.sequential_index = 0

   def random_batch(self, batch_size):
       choose = random.sample(list(range(0, self.train_num_data)), batch_size)
       x = self.train_x[choose]
       y = self.train_y[choose]
       patient_id = np.array(self.train_patient_id)[choose]
       return x, y, patient_id


   def sequential_batch(self):
       x = self.test_x[self.sequential_index]
       patient_id = self.test_patient_id[self.sequential_index]
       temp_list = self.test_time_stamp.append()
       self.sequential_index += 1
       if (self.sequential_index >= self.test_num_data):
           return x, patient_id, True
       return x, patient_id, False




if __name__ == '__main__':
    data = DataSet('../_data/ml_10_medicalalert_train.csv', '../_data/ml_10_medicalalert_test.csv')
    print(data.random_batch(16))
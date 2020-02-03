import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

desired_width = 320
pd.set_option('display.width', desired_width)
pd.set_option('display.max_columns', 20)

#loading data
train = pd.read_csv("C:/Users/lg/Desktop/skcc/skcc_4_1_ML/_data/ml_7_medicalalert_train.csv", encoding = 'utf-8')
test = pd.read_csv("C:/Users/lg/Desktop/skcc/skcc_4_1_ML/_data/ml_7_medicalalert_test.csv", encoding = 'utf-8')

#sorting of patientid and time
train.sort_values(by=['patientid','timestamp'])

#change the object value to flaot
#train.astype({'w':'float', 'pw':'float', "bps":'float', 'bpd':'float', 'spo2':'float', 'hr':'float','gl':'float'})

#sepeartae sex
train_man = train[train['gender'] =='M']
train_girl = train[train['gender'] =='F']

#summary
print(train_man.describe())
print(train_girl.describe())

#replace outlier to Nan value
outlier_value = ['w', 'pw', 'bps', 'spo2', 'hr', 'gl']
def std_based_outlier(df):
    for col in outlier_value:
        low = df[col].quantile(.05)
        high =df[col].quantile(.95)

        for i in df.index:
            val = df.get_value(i, col)
            if val < low or val > high or val ==0 :
                df.set_value(i, col, np.nan)

std_based_outlier(train_man);
std_based_outlier(train_girl);

########################################################################################################################

train_man_y = train_man[train_man["alert"] == 'Yes']
train_man_n = train_man[train_man["alert"] == 'No']

plt.figure(figsize=(7, 6)) # 크기 지정
boxplot = train_man.dropna().boxplot(column=outlier_value, by='alert')
plt.yticks(np.arange(40, 250, step=15))
plt.show()


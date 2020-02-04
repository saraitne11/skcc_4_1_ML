import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

desired_width = 320
pd.set_option('display.width', desired_width)
pd.set_option('display.max_columns', 20)


#loading data
train = pd.read_csv("../_data/ml_7_medicalalert_traincsv", encoding = 'utf-8')
test = pd.read_csv( "../_data/ml_7_medicalalert_test.csv", encoding = 'utf-8')



#sorting of patientid and time
train = train.sort_values(by=['patientid','timestamp'])

#change the object value to flaot
#train.astype({'w':'float', 'pw':'float', "bps":'float', 'bpd':'float', 'spo2':'float', 'hr':'float','gl':'float'})

#sepeartae sex
train_man = train[train['gender'] =='M']
train_girl = train[train['gender'] =='F']

#summary
# print(train_man.describe())
# print(train_girl.describe())

#replace outlier to Nan value
outlier_value = ['w', 'pw', 'bps', 'bpd', 'spo2', 'hr', 'gl']
def remove_outlier(df):
    for col in outlier_value:
        low = df[col].quantile(.05)
        high =df[col].quantile(.95)
        for i in df.index:
            val = df.get_value(i, col)
            if val < low or val > high or val ==0 :
                df.set_value(i, col, np.nan)

remove_outlier(train_man);
remove_outlier(train_girl);
# df.col = df.groupby('patientid')[col].apply(lambda x: x.fillna(x.mean()))
# df.col = df.col.fillna(df.col.mean())

# fill the Nan value
# def fillNa(df):
#     for col in outlier_value:
#         df[col] = df.groupby('patientid')[col].apply(lambda x: x.fillna(x.mean()))
#         df[col] = df[col].fillna(df[col].mean())
#
#
# fillNa(train_man)

# #value Normalization
# def normalizaiton(df):
#     for col in outlier_value:
#         df[col] = (df[col] - df[col].min()) / (df[col].max()-df[col].min())
#
# normalizaiton(train_man);




#print(train_man)
########################################################################################################################

## alert Y and N, box plot
# train_man_y = train_man[train_man["alert"] == 'Yes']
# train_man_n = train_man[train_man["alert"] == 'No']
#
# plt.figure(figsize=(7, 6)) # 크기 지정
# boxplot = train_man.dropna().boxplot(column=outlier_value, by='alert')
# plt.yticks(np.arange(40, 250, step=15))
# plt.show()
# #
# print(train_man.descibe())
# print(train_man.groupby('patientid').std())

# show correaion heat map plot
# sns.heatmap(train_man.corr(), annot = True, fmt='.1g')
# plt.show()

## converting serval rows into one row
# train_man = train_man.set_index(['patientid', train_man.groupby(['patientid']).cumcount()+1]).unstack().sort_index(level=1, axis=1)
# train_man.columns = train_man.columns.map('{0[0]}_{0[1]}'.format)
# train_man.reset_index()


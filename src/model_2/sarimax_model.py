#-*- coding:utf-8 -*-#
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import statsmodels.api as sm
import pickle
from sklearn.externals import joblib
from os import path, listdir, mkdir
import warnings
warnings.filterwarnings("ignore")

def sarimax_model_build():

    # 外部特征
    data = ['T1D0'] #, 'T1D1', 'T2D0', 'T3D0', 'T3D1'
    peak = ['EarlyPeakTime', 'LatePeakTime', 'NormalTime']
    holiday = ['BeforeNationalDay', 'NationalDayStart', 'NationalDayEnd', 'Weekend', 'WorkingDay', 'WorkingWeekend']
    weather = ['BigRain', 'MediumRain', 'SmallRain', 'Sunny']
    train_data = pd.read_csv('sarimax_data.csv',index_col=0)
    train_data[weather] = train_data[weather].fillna(0)
    order = (6, 0, 1)
    seasonal_order = (1, 0, 1, 72)

    out_model_pkl = 'SARIMAX_%d_%d_%d_%d_%d_%d_%d_%%s.pkl' %(order+seasonal_order)
    out_model_path = '../../data/data_after_process/tmp_file'

    # if out_model_path not in listdir(out_model_path):
    #     mkdir(out_model_path)

    for i in data:

        print i + ' model start training!'
        td_data =  train_data[i]
        td_data_log = td_data.map(lambda x:np.log(x+1))
        mod = sm.tsa.statespace.SARIMAX(
            endog=td_data_log,
            exog=train_data[holiday+weather+peak],
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        answer = mod.fit()
        # print(results.summary())
        with open(path.join(out_model_path,out_model_pkl%(i)),'wb') as model_file:
            pickle.dump(answer, model_file, -1)

'''
时间序列模型建模
'''

if __name__ == '__main__':
    sarimax_model_build()


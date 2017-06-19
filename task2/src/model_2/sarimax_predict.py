#-*- coding:utf-8 -*-#
import pickle
import pandas as pd
import numpy as np
import datetime
from os import path
from pandas.tseries.offsets import Minute, Hour, Day
from datetime import timedelta
from sklearn.externals import joblib

def sarimax_predict():

    model, prediction = {}, {}
    data = ['T1D0'] #, 'T1D1', 'T2D0', 'T3D0', 'T3D1'
    train_data = pd.read_csv('sarimax_data.csv',index_col=0)
    in_model_pkl = 'SARIMAX_6_0_1_1_0_1_72_%s.pkl'
    in_model_path = '../../data/data_after_process/tmp_file'

    for i in data:
        model[i] = joblib.load(path.join(in_model_path, in_model_pkl % (i)))
        # print results[td].summary()
        print i + ' model start predicting!'
        prediction[i] = model[i].predict(0, len(train_data)-1)
        prediction[i] = prediction[i].map(lambda x: np.round(np.exp(x) - 1, 2))

    answer = pd.DataFrame(prediction)['2016-10-25':]
    answer = pd.concat([
        answer.between_time('17:00', '18:40'), answer.between_time('8:00', '9:40')
    ]).sort_index()
    answer['time_window'] = answer.index.map(lambda x:'['+str(x)+','+str(x+Minute(20))+')')
    answer = pd.melt(
        answer,
        var_name='tollgate_id',
        value_name='volume',
        id_vars=['time_window'])

    answer['direction'] = answer['tollgate_id'].map(lambda d: int(d[3]))
    answer['tollgate_id'] = answer['tollgate_id'].map(lambda d: int(d[1]))
    answer = answer[['tollgate_id','time_window','direction','volume']]

    # import time
    # version = time.strftime('%Y-%m-%d_%R', time.localtime(time.time()))
    # answer.to_csv('answer/prediction_'+version+'.csv',float_format='%.2f',header=True,index=False,encoding='utf-8')
    answer.to_csv('../../answer/prediction_sarimax.csv',float_format='%.2f',header=True,index=False,encoding='utf-8')

'''
时间序列模型预测
'''

if __name__ == 'main':
    sarimax_predict()
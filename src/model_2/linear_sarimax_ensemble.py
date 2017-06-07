#-*- coding:utf-8 -*-#
import pandas as pd
import numpy as np
import datetime
from os import path
from pandas.tseries.offsets import Minute, Hour, Day

def linear_sarimax_ensemble():
    # id 2, direction 0 use sarimax's prediction, others use linear_regression's prediction
    linear_regression_file_name = '../../answer/prediction_linear_regression.csv'
    sarimax_file_name = '../../answer/prediction_sarimax.csv'
    linear_regression_result = pd.read_csv(linear_regression_file_name)
    sarimax_result = pd.read_csv(sarimax_file_name)
    linear_regression_result_apart_1_0 = linear_regression_result[(linear_regression_result['tollgate_id'] == 2)|(linear_regression_result['tollgate_id'] == 3)|
                                      ((linear_regression_result['tollgate_id'] == 1)&(linear_regression_result['direction'] == 1))]
    # sarimax_result_1_0 = sarimax_result[(sarimax_result['tollgate_id'] == 1)&(sarimax_result['direction'] == 0)]
    ensemble_result = pd.concat((sarimax_result, linear_regression_result_apart_1_0), ignore_index=True)
    ensemble_result.to_csv('../../answer/linear_sarimax_ensemble.csv',float_format='%.2f',header=True,index=False,encoding='utf-8')

'''
模型融合
'''

if __name__ == '__main__':
    linear_sarimax_ensemble()
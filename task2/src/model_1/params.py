# -*- coding:utf8 -*-#
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from dateutil.parser import parse
import warnings

warnings.filterwarnings("ignore")
from sklearn.linear_model import LinearRegression
from sklearn.svm import LinearSVR, NuSVR
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, ShuffleSplit
from sklearn.metrics import make_scorer
from pandas.tseries.offsets import Minute, Hour

'''
损失函数定义
'''
def mape_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true))


def scoring(reg, x, y):
    pred = reg.predict(x)
    return -mape_error(pred, y)


'''
网格化参数设定
'''
params = {'n_estimators': [10, 20, 30, 40, 50, 80, 100], 'max_depth': [2, 5, 8, 10, 15],
          'min_samples_split': [2, 5, 10, 15], 'min_samples_leaf': [1, 2, 5, 10, 15],
          'max_features': ['auto', 'sqrt', 'log2', None]}

EXT = ExtraTreesRegressor(n_jobs=-1)
cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=2)
CLF = GridSearchCV(EXT, params, scoring=scoring, verbose=6, cv=cv)

'''
参数训练
'''
def param(x1_file, y1_file, params_file):
    X1 = np.loadtxt(x1_file, delimiter=",")
    Y1 = np.loadtxt(y1_file, delimiter=",")

    CLF.fit(X1, Y1)
    print 'best params:\n', CLF.best_params_

    mean_scores = np.array(CLF.cv_results_['mean_test_score'])
    print 'mean score', mean_scores
    print 'best score', CLF.best_score_
    fr_to = open(params_file, "w")
    fr_to.write(str(CLF.best_params_) + "\n")
    fr_to.write(str(mean_scores) + "\n")
    fr_to.write(str(CLF.best_score_) + "\n")


def main():
    jiange = 20
    afterNoon = "6_7"
    afterNoon_ = "8_9"
    x1_file = '../../data/data_after_process/train_test/train_' + afterNoon + '_' + str(jiange) + '_feature.csv'
    y1_file = '../../data/data_after_process/train_test/train_' + afterNoon_ + '_' + str(jiange) + '_y.csv'
    params_file = "../../data/params1_mean_2.txt"
    param(x1_file, y1_file, params_file)

    afterNoon = "15_16"
    afterNoon_ = "17_18"
    x2_file = '../../data/data_after_process/train_test/train_' + afterNoon + '_' + str(jiange) + '_feature.csv'
    y2_file = '../../data/data_after_process/train_test/train_' + afterNoon_ + '_' + str(jiange) + '_y.csv'
    params_file = "../../data/params2_mean_2.txt"
    param(x2_file, y2_file, params_file)

'''
利用网格化搜索学习参数，选出最有的参数
大概需要三小时来训练参数
'''

if __name__ == '__main__':
    main()

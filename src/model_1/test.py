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
from feature import data_fliter

'''
最后一周作为测试数据
'''
slpit_array = ['10-18', '10-19', '10-20', '10-21', '10-22', '10-23', '10-24']

def scoring(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true))


def build_time_window(day, hour, minute):
    time_window = datetime(2016, 10, day, hour, minute, 0)
    return '[' + str(time_window) + ',' + str(time_window + Minute(20)) + ')'


def get_test_y(filename, test_list):
    a = np.loadtxt(filename, delimiter=",")
    a = data_fliter(a)
    test_y = a[test_list]
    test_y = test_y[:, -6:]
    return test_y


def get_testIdList(filename):
    a = np.loadtxt(filename, delimiter=",")
    a = data_fliter(a)
    test_list = []
    train_list = []
    for i in range(a.shape[0]):
        id = int(a[i, 0])
        direction = int(a[i, 1])
        month = int(a[i, 2])
        day = int(a[i, 3])
        week = int(a[i, 4])
        month_day = "%02d-%02d" % (month, day)
        if month_day in slpit_array:
            # print i
            test_list.append(i)
        else:
            train_list.append(i)
    # print train_list, test_list
    return train_list, test_list, a[test_list]


def split_train_val(X1, Y1, train_list, test_list):
    train_X = X1[train_list]
    train_y = Y1[train_list]
    test_X = X1[test_list]
    test_Y = Y1[test_list]
    return train_X, train_y, test_X, test_Y


def test(x_file, y_file, train_list, test_list, best_params):
    X1 = np.loadtxt(x_file, delimiter=",")
    Y1 = np.loadtxt(y_file, delimiter=",")
    train_X, train_Y, test_X, _ = split_train_val(X1, Y1, train_list, test_list)
    # print train_X.shape,test_X.shape
    EXT1 = ExtraTreesRegressor(n_jobs=-1, random_state=1, **best_params)
    EXT1.fit(train_X, train_Y)
    test_Y1 = EXT1.predict(test_X)
    # print EXT1.feature_importances_
    return test_Y1


def get_ans(test_y_true, test_Y, ratio_file):
    import cPickle
    all_ratio = cPickle.load(open(ratio_file, "rb"))
    # print all_ratio
    ans = []
    for line, y_sum in zip(test_y_true, test_Y):
        id = int(line[0])
        direction = int(line[1])
        month = int(line[2])
        day = int(line[3])
        week = int(line[4])
        ratio = all_ratio[id][direction][week]
        tmp = np.hstack((ratio[:3] * y_sum[0], ratio[3:] * y_sum[1]))
        # print ratio, tmp
        ans.append(tmp)
    return np.array(ans)

def main():
    '''
    上午的模型结果
    '''
    best_params1 = {'max_features': 'log2', 'min_samples_split': 2, 'n_estimators': 30, 'max_depth': 15,
                    'min_samples_leaf': 2}

    jiange = 20
    afterNoon = "6_7"
    afterNoon_ = "8_9"
    filename = '../../data/data_after_process/train_' + afterNoon + '_' + str(jiange) + '.csv'
    train_list, test_list, test_y_true = get_testIdList(filename)
    filename1 = '../../data/data_after_process/train_' + afterNoon_ + '_' + str(jiange) + '.csv'
    test_y = get_test_y(filename1, test_list)
    # print test_y.shape
    x1_file = '../../data/data_after_process/train_test/train_' + afterNoon + '_' + str(jiange) + '_feature.csv'
    y1_file = '../../data/data_after_process/train_test/train_' + afterNoon_ + '_' + str(jiange) + '_y.csv'
    test1_file = '../../data/data_after_process/train_test/test_' + afterNoon + '_' + str(jiange) + '_feature.csv'
    ratio_file = '../../data/data_after_process/train_test/train_' + afterNoon_ + '_' + str(jiange) + '_ratio.pkl'
    test_file_orign = '../../data/data_after_process/test_' + afterNoon + '_' + str(jiange) + '.csv'
    test_Y1 = test(x1_file, y1_file, train_list, test_list, best_params1)
    ans1 = get_ans(test_y_true, test_Y1, ratio_file)
    print '上午结果：',scoring(test_y, ans1)


    '''
    下午的模型结果
    '''
    best_params2 = {'max_features': 'auto', 'min_samples_split': 15, 'n_estimators': 10, 'max_depth': 15,
                    'min_samples_leaf': 1}

    afterNoon = "15_16"
    afterNoon_ = "17_18"
    filename = '../../data/data_after_process/train_' + afterNoon + '_' + str(jiange) + '.csv'
    train_list, test_list, test_y_true = get_testIdList(filename)
    # print test_list
    filename1 = '../../data/data_after_process/train_' + afterNoon_ + '_' + str(jiange) + '.csv'
    test_y = get_test_y(filename1, test_list)
    x2_file = '../../data/data_after_process/train_test/train_' + afterNoon + '_' + str(jiange) + '_feature.csv'
    y2_file = '../../data/data_after_process/train_test/train_' + afterNoon_ + '_' + str(jiange) + '_y.csv'
    test2_file = '../../data/data_after_process/train_test/test_' + afterNoon + '_' + str(jiange) + '_feature.csv'
    ratio_file = '../../data/data_after_process/train_test/train_' + afterNoon_ + '_' + str(jiange) + '_ratio.pkl'
    test_file_orign = '../../data/data_after_process/test_' + afterNoon + '_' + str(jiange) + '.csv'
    test_Y2 = test(x2_file, y2_file, train_list, test_list, best_params2)
    ans2 = get_ans(test_y_true, test_Y2, ratio_file)
    # print test_y
    # print ans2
    print '下午结果：',scoring(test_y, ans2)


'''
用最后一周作为测试数据来看看模型的结果
'''

if __name__ == '__main__':
    main()
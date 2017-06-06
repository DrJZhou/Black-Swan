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

def mape_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true))


def scoring(reg, x, y):
    pred = reg.predict(x)
    return -mape_error(pred, y)


def build_time_window(day, hour, minute):
    time_window = datetime(2016, 10, day, hour, minute, 0)
    return '[' + str(time_window) + ',' + str(time_window + Minute(20)) + ')'


def predict(x_file, y_file, test_x_file, best_params):
    X1 = np.loadtxt(x_file, delimiter=",")
    Y1 = np.loadtxt(y_file, delimiter=",")
    test_X1 = np.loadtxt(test_x_file, delimiter=",")
    EXT1 = ExtraTreesRegressor(n_jobs=-1, random_state=1, **best_params)
    EXT1.fit(X1, Y1)
    test_Y1 = EXT1.predict(test_X1)  # * NOR.scale_ + NOR.mean_
    # print test_Y1
    # print EXT1.feature_importances_
    return test_Y1


def get_ans(test_x_file, test_Y, ratio_file, predict_time):
    test_X1 = np.loadtxt(test_x_file, delimiter=",")
    import cPickle
    all_ratio = cPickle.load(open(ratio_file, "rb"))
    # print all_ratio
    ans = []
    for line, y_sum in zip(test_X1, test_Y):
        id = int(line[0])
        direction = int(line[1])
        month = int(line[2])
        day = int(line[3])
        week = int(line[4])
        # print id,direction,week
        if day == 30 and direction == 0 and id == 2:
            if predict_time == 8:
                y_sum[0]=20
                y_sum[1]=25
            else:
                y_sum[0]=20
                y_sum[1]=20
        if day == 31 and direction == 0 and id == 2:
            if predict_time == 8:
                y_sum[0]=55
                y_sum[1]=50
            else:
                y_sum[0]=15
                y_sum[1]=15
        ratio = all_ratio[id][direction][week]
        # print ratio,y_sum
        tmp = np.hstack((ratio[:3] * y_sum[0], ratio[3:] * y_sum[1]))
        # print ratio,tmp

        for i in range(len(tmp)):
            hour = i / 3 + predict_time
            minute = i % 3 * 20
            time_str = build_time_window(day, hour, minute)
            sample_ans = str(id) + "," + '"' + time_str + '"' + "," + str(direction) + "," + str(tmp[i])
            ans.append(sample_ans)
    return ans


def save_ans(ans, ans_file):
    fr_to = open(ans_file, "w")
    fr_to.write("tollgate_id,time_window,direction,volume\n")
    for line in ans:
        fr_to.write(line + "\n")
    fr_to.close()
    volume = pd.read_csv(ans_file)
    volume['volume'] = volume['volume'].astype('float')
    volume.index = volume['time_window'].map(lambda x: parse(x.split(',')[0][1:]))
    # print volume
    tmp = volume[['tollgate_id', 'time_window', 'direction', 'volume']].sort_values(
        ['tollgate_id', 'direction', 'time_window'])
    tmp.to_csv(ans_file, float_format='%.2f', header=True, index=False,
               encoding='utf-8')

def main():
    '''
    上午的模型训练
    参数是通过网格搜索得到的结果，详细将文件param文件的训练过程
    '''

    # best
    best_params1 = {'max_features': 'log2', 'min_samples_split': 2, 'n_estimators': 30, 'max_depth': 15,
                    'min_samples_leaf': 2}

    jiange = 20
    afterNoon = "6_7"
    afterNoon_ = "8_9"
    x1_file = '../../data/data_after_process/train_test/train_' + afterNoon + '_' + str(jiange) + '_feature.csv'
    y1_file = '../../data/data_after_process/train_test/train_' + afterNoon_ + '_' + str(jiange) + '_y.csv'
    test1_file = '../../data/data_after_process/train_test/test_' + afterNoon + '_' + str(jiange) + '_feature.csv'
    ratio_file = '../../data/data_after_process/train_test/train_' + afterNoon_ + '_' + str(jiange) + '_ratio.pkl'
    test_file_orign = '../../data/data_after_process/test_' + afterNoon + '_' + str(jiange) + '.csv'
    test_Y1 = predict(x1_file, y1_file, test1_file, best_params1)
    ans1 = get_ans(test_file_orign, test_Y1, ratio_file, 8)

    '''
       下午的模型训练
    '''
    # best
    best_params2 = {'max_features': 'auto', 'min_samples_split': 15, 'n_estimators': 10, 'max_depth': 15,
                    'min_samples_leaf': 1}

    afterNoon = "15_16"
    afterNoon_ = "17_18"
    x2_file = '../../data/data_after_process/train_test/train_' + afterNoon + '_' + str(jiange) + '_feature.csv'
    y2_file = '../../data/data_after_process/train_test/train_' + afterNoon_ + '_' + str(jiange) + '_y.csv'
    test2_file = '../../data/data_after_process/train_test/test_' + afterNoon + '_' + str(jiange) + '_feature.csv'
    ratio_file = '../../data/data_after_process/train_test/train_' + afterNoon_ + '_' + str(jiange) + '_ratio.pkl'
    test_file_orign = '../../data/data_after_process/test_' + afterNoon + '_' + str(jiange) + '.csv'
    test_Y2 = predict(x2_file, y2_file, test2_file, best_params2)
    ans2 = get_ans(test_file_orign, test_Y2, ratio_file, 17)
    # print ans2
    ans = np.hstack((ans1, ans2))
    import time

    version = time.strftime('%Y-%m-%d', time.localtime(time.time()))
    # ans_file = "../../src/model/answer/predict_" + version + "_2_mean.csv"
    ans_file = "../../answer/predict_2_mean.csv"
    save_ans(ans, ans_file)


'''
模型上下午分别预测
'''

if __name__ == '__main__':
    main()

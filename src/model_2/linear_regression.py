# -*- coding:utf8 -*-#
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from matplotlib import pyplot as plt
# import seaborn as sns
from dateutil.parser import parse
from pandas.tseries.offsets import Minute, Hour, Day
import warnings

warnings.filterwarnings("ignore")

params = {1: [0.04, 0., 0.36, 0.58], 2: [0.37, 0.17, 0.27, 0.22], 3: [0.33, 0.1, 0.09, 0.47],
          4: [0.03, 0.41, 0.37, 0.15]}


def mape_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true))


def data_preprocess(volume_file, volume_file_new, test_volume_file):
    volume = pd.read_csv(volume_file)
    volume_new = pd.read_csv(volume_file_new)
    test_volume = pd.read_csv(test_volume_file)

    time_window = pd.date_range(start=datetime(2016, 9, 19), end=datetime(2016, 10, 18) + Day(7), freq='20min',
                                closed='left').map(lambda x: '[' + str(x) + ',' + str(x + Minute(20)) + ')')
    fill_null_dataframe = pd.DataFrame(
        {'tollgate_id': len(time_window) * 2 * [1] + len(time_window) * [2] + len(time_window) * 2 * [3],
         'direction': len(time_window) * [0] + len(time_window) * [1] + len(time_window) * [0] + len(time_window) * [
             0] + len(time_window) * [1],
         'time_window': np.tile(time_window, 5)})
    volume = pd.concat((volume, volume_new), ignore_index=True)
    volume = pd.merge(volume, fill_null_dataframe, how='right').fillna(0).sort_values(
        ['tollgate_id', 'direction', 'time_window'])  # use fill null values
    volume['volume'] = volume['volume'].astype('float')
    volume['tollgate_id'] = volume['tollgate_id'].astype('int')
    volume['direction'] = volume['direction'].astype('int')
    volume.index = volume['time_window'].map(lambda x: parse(x.split(',')[0][1:]))
    volume.drop(pd.date_range(start=datetime(2016, 10, 1), end=datetime(2016, 10, 8), freq='10min', closed='left'),
                inplace=True, errors='ignore')
    volume.index = volume.index.map(lambda x: x + Day(7) if datetime(2016, 9, 19) <= x < datetime(2016, 10, 1) else x)
    volume['time_window_weekday'] = volume.index.map(lambda x: x.weekday)
    volume['time_window_month'] = volume.index.map(lambda x: x.month)
    volume['time_window_day'] = volume.index.map(lambda x: x.day)
    volume['time_window_hour'] = volume.index.map(lambda x: x.hour)
    volume['time_window_minute'] = volume.index.map(lambda x: x.minute)
    volume['time_window_week'] = -1
    volume['time_window_week'][(volume['time_window_day'] > 26) | (volume['time_window_day'] < 4)] = 0
    volume['time_window_week'][(volume['time_window_day'] >= 4) & (volume['time_window_day'] < 11)] = 1
    volume['time_window_week'][(volume['time_window_day'] >= 11) & (volume['time_window_day'] < 18)] = 2
    volume['time_window_week'][(volume['time_window_day'] >= 18) & (volume['time_window_day'] < 25)] = 3

    test_volume['volume'] = test_volume['volume'].astype('float')
    test_volume['tollgate_id'] = test_volume['tollgate_id'].astype('int')
    test_volume['direction'] = test_volume['direction'].astype('int')
    test_volume.index = test_volume['time_window'].map(lambda x: parse(x.split(',')[0][1:]))
    test_volume['time_window_weekday'] = test_volume.index.map(lambda x: x.weekday)
    test_volume['time_window_month'] = test_volume.index.map(lambda x: x.month)
    test_volume['time_window_day'] = test_volume.index.map(lambda x: x.day)
    test_volume['time_window_hour'] = test_volume.index.map(lambda x: x.hour)
    test_volume['time_window_minute'] = test_volume.index.map(lambda x: x.minute)
    return volume, test_volume


def linear_regression_main(volume, test_volume, predict_from_params):
    res = pd.DataFrame()
    for i in xrange(3):
        for j in xrange(2):
            if not (i + 1 == 2 and j == 1):
                # extract four weeks volume
                volume_6_7_15_16 = volume[(volume['tollgate_id'] == i + 1) & (volume['direction'] == j) & (
                    (volume['time_window_hour'] == 6) | (volume['time_window_hour'] == 7) | (
                        volume['time_window_hour'] == 15) | (volume['time_window_hour'] == 16))]
                week_volume_0 = volume_6_7_15_16[volume_6_7_15_16['time_window_week'] == 0]['volume'].values
                week_volume_1 = volume_6_7_15_16[volume_6_7_15_16['time_window_week'] == 1]['volume'].values
                week_volume_2 = volume_6_7_15_16[volume_6_7_15_16['time_window_week'] == 2]['volume'].values
                week_volume_3 = volume_6_7_15_16[volume_6_7_15_16['time_window_week'] == 3]['volume'].values
                test_volume_tmp = test_volume[
                    (test_volume['tollgate_id'] == i + 1) & (test_volume['direction'] == j) & (
                        (test_volume['time_window_hour'] == 6) | (test_volume['time_window_hour'] == 7) | (
                            test_volume['time_window_hour'] == 15) | (test_volume['time_window_hour'] == 16))][
                    'volume'].values
                # id 2, direction 0 need treat specially
                if i + 1 == 2:
                    min_loss_weekend, weight_array_weekend = float('inf'), np.zeros(4)
                    min_loss_no_weekend, weight_array_no_weekend = float('inf'), np.zeros(4)

                    week_volume_0_no_weekend = week_volume_0[:5 * 12]
                    week_volume_0_weekend = week_volume_0[5 * 12:]
                    week_volume_1_no_weekend = week_volume_1[:5 * 12]
                    week_volume_1_weekend = week_volume_1[5 * 12:]
                    week_volume_2_no_weekend = week_volume_2[:5 * 12]
                    week_volume_2_weekend = week_volume_2[5 * 12:]
                    week_volume_3_no_weekend = week_volume_3[:5 * 12]
                    week_volume_3_weekend = week_volume_3[5 * 12:]
                    test_week_volume_no_weekend = test_volume_tmp[:5 * 12]
                    test_week_volume_weekend = test_volume_tmp[5 * 12:]
                    print predict_from_params
                    if not predict_from_params:
                        # train phase
                        for a in np.linspace(0,1,101):
                            for b in np.linspace(0,1,101):
                                for c in np.linspace(0,1,101):
                                    for d in np.linspace(0,1,101):
                                        loss_weekend = mape_error(test_week_volume_weekend, (a * week_volume_0_weekend + b * week_volume_1_weekend + c * week_volume_2_weekend + d * week_volume_3_weekend ))
                                        loss_no_weekend = mape_error(test_week_volume_no_weekend, (a * week_volume_0_no_weekend + b * week_volume_1_no_weekend + c * week_volume_2_no_weekend + d * week_volume_3_no_weekend ))
                                        if loss_weekend < min_loss_weekend:
                                            min_loss_weekend = loss_weekend
                                            weight_array_weekend = np.array([a, b, c, d])
                                        if loss_no_weekend < min_loss_no_weekend:
                                            min_loss_no_weekend = loss_no_weekend
                                            weight_array_no_weekend = np.array([a, b, c, d])

                    else:
                        weight_array_weekend, weight_array_no_weekend = [ 0.09, 0., 0., 0.14], [ 0.32, 0., 0.21, 0.46]

                    volume_8_9_17_18 = volume[(volume['tollgate_id'] == i + 1) & (volume['direction'] == j) & (
                        (volume['time_window_hour'] == 8) | (volume['time_window_hour'] == 9) | (
                            volume['time_window_hour'] == 17) | (volume['time_window_hour'] == 18))]

                    week_volume_0 = volume_8_9_17_18[volume_8_9_17_18['time_window_week'] == 0]['volume'].values
                    week_volume_1 = volume_8_9_17_18[volume_8_9_17_18['time_window_week'] == 1]['volume'].values
                    week_volume_2 = volume_8_9_17_18[volume_8_9_17_18['time_window_week'] == 2]['volume'].values
                    week_volume_3 = volume_8_9_17_18[volume_8_9_17_18['time_window_week'] == 3]['volume'].values

                    week_volume_0_no_weekend = week_volume_0[:5 * 12]
                    week_volume_0_weekend = week_volume_0[5 * 12:]
                    week_volume_1_no_weekend = week_volume_1[:5 * 12]
                    week_volume_1_weekend = week_volume_1[5 * 12:]
                    week_volume_2_no_weekend = week_volume_2[:5 * 12]
                    week_volume_2_weekend = week_volume_2[5 * 12:]
                    week_volume_3_no_weekend = week_volume_3[:5 * 12]
                    week_volume_3_weekend = week_volume_3[5 * 12:]
                    
                    # predict phase
                    print i + 1, j, weight_array_weekend, weight_array_no_weekend
                    tmp_volume_weekend = weight_array_weekend[0] * week_volume_0_weekend + weight_array_weekend[
                                                                                               1] * week_volume_1_weekend + \
                                         weight_array_weekend[2] * week_volume_2_weekend + weight_array_weekend[
                                                                                               3] * week_volume_3_weekend
                    tmp_volume_no_weekend = weight_array_no_weekend[0] * week_volume_0_no_weekend + \
                                            weight_array_no_weekend[
                                                1] * week_volume_1_no_weekend + \
                                            weight_array_no_weekend[2] * week_volume_2_no_weekend + \
                                            weight_array_no_weekend[
                                                3] * week_volume_3_no_weekend
                    tmp_res = pd.DataFrame({'tollgate_id': i + 1, 'direction': j,
                                            'time_window': [datetime(2016, 10, day, hour, minute) for day in
                                                            range(25, 30) for hour in [8, 9, 17, 18] for minute in
                                                            [0, 20, 40]],
                                            'volume': tmp_volume_no_weekend})
                    res = pd.concat((res, tmp_res))
                    tmp_res = pd.DataFrame({'tollgate_id': i + 1, 'direction': j,
                                            'time_window': [datetime(2016, 10, day, hour, minute) for day in
                                                            range(30, 32) for hour in [8, 9, 17, 18] for minute in
                                                            [0, 20, 40]],
                                            'volume': tmp_volume_weekend})
                    res = pd.concat((res, tmp_res))

                else:
                    min_loss, weight_array = float('inf'), np.zeros(4)

                    if not predict_from_params:
                        # train phase
                        for a in np.linspace(0,1,101):
                            for b in np.linspace(0,1,101):
                                for c in np.linspace(0,1,101):
                                    for d in np.linspace(0,1,101):
                                        loss = mape_error(test_volume_tmp, (a * week_volume_0 + b * week_volume_1 + c * week_volume_2 + d * week_volume_3))
                                        if loss < min_loss:
                                            min_loss = loss
                                            weight_array = np.array([a, b, c, d])

                    else:
                        weight_array = params[i + 1 + j]

                    volume_8_9_17_18 = volume[(volume['tollgate_id'] == i + 1) & (volume['direction'] == j) & (
                        (volume['time_window_hour'] == 8) | (volume['time_window_hour'] == 9) | (
                            volume['time_window_hour'] == 17) | (volume['time_window_hour'] == 18))]
                    # predict phase
                    week_volume_0 = volume_8_9_17_18[volume_8_9_17_18['time_window_week'] == 0]['volume'].values
                    week_volume_1 = volume_8_9_17_18[volume_8_9_17_18['time_window_week'] == 1]['volume'].values
                    week_volume_2 = volume_8_9_17_18[volume_8_9_17_18['time_window_week'] == 2]['volume'].values
                    week_volume_3 = volume_8_9_17_18[volume_8_9_17_18['time_window_week'] == 3]['volume'].values
                    print i + 1, j, weight_array
                    tmp_volume = weight_array[0] * week_volume_0 + weight_array[1] * week_volume_1 + weight_array[
                                                                                                         2] * week_volume_2 + \
                                 weight_array[3] * week_volume_3
                    tmp_res = pd.DataFrame({'tollgate_id': i + 1, 'direction': j,
                                            'time_window': [datetime(2016, 10, day, hour, minute) for day in
                                                            range(18 + 7, 25 + 7) for hour in [8, 9, 17, 18] for minute
                                                            in [0, 20, 40]],
                                            'volume': tmp_volume})
                    res = pd.concat((res, tmp_res))

    answer = res[['tollgate_id', 'time_window', 'direction', 'volume']]
    answer['time_window'] = answer['time_window'].map(lambda x: '[' + str(x) + ',' + str(x + Minute(20)) + ')')
    # import time
    # version = time.strftime('%Y-%m-%d_%R', time.localtime(time.time()))
    # res.to_csv('answer/prediction_'+version+'.csv',float_format='%.2f',header=True,index=False,encoding='utf-8')
    answer.to_csv('../../answer/prediction_linear.csv', float_format='%.2f', header=True, index=False, encoding='utf-8')

# predict_from_params for predicting with no training
def main(predict_from_params = True):
    basepath = '../../data/data_after_process/'
    volume_file = basepath + 'training_20min_avg_volume.csv'
    volume_file_new = basepath + 'training2_20min_avg_volume.csv'
    test_volume_file = basepath + 'test2_20min_avg_volume.csv'
    volume, test_volume = data_preprocess(volume_file, volume_file_new, test_volume_file)
    print 'knn is running!'
    linear_regression_main(volume, test_volume, predict_from_params)


if __name__ == '__main__':
    main()

    # 1 0 [ 0.04  0.    0.36  0.58]
    # 1 1 [ 0.37  0.17  0.27  0.22]
    # 2 0 [ 0.09  0.    0.    0.14] [ 0.32  0.    0.21  0.46]
    # 3 0 [ 0.33  0.1   0.09  0.47]
    # 3 1 [ 0.03  0.41  0.37  0.15]

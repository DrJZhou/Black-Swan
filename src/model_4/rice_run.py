#-*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.svm import LinearSVR
import lightgbm as lgb
import xgboost as xgb

VOLUME_TABLE6_TRAIN = "../../data/dataset/volume(table 6)_training.csv"
VOLUME_TABLE6_TRAIN2 = "../../data/dataset/volume(table 6)_training2.csv"
VOLUME_TABLE6_TEST2 = "../../data/dataset/volume(table 6)_test2.csv"

WEATHER_TABLE7_UPDATE = "../../data/dataset/weather (table 7)_training_update.csv"
WEATHER_TABLE7_test1 = "../../data/dataset/weather (table 7)_test1.csv"
WEATHER_TABLE7_test2 = "../../data/dataset/weather (table 7)_2.csv"



def load_data(train_path1, train_path2, test_path, weather_path_update, weather_path_test1, weather_path_test2):
    df_train1 = pd.read_csv(train_path1, header=0)
    df_train2 = pd.read_csv(train_path2, header=0)
    df_train1.columns = ["time", "tollgate_id", "direction", "vehicle_model", "has_etc", "vehicle_type"]
    df_train2.columns = ["time", "tollgate_id", "direction", "vehicle_model", "has_etc", "vehicle_type"]
    df_train = pd.concat([df_train1, df_train2], ignore_index=True)
    print("train1", df_train1.shape)
    print("train2", df_train2.shape)
    print("train", df_train.shape)

    df_test = pd.read_csv(test_path, header=0)
    df_test.columns = ["time", "tollgate_id", "direction", "vehicle_model", "has_etc", "vehicle_type"]
    df_weather_update = pd.read_csv(weather_path_update, header=0)
    df_weather_test1 = pd.read_csv(weather_path_test1, header=0)
    df_weather_test2 = pd.read_csv(weather_path_test2, header=0)
    df_weather = pd.concat([df_weather_update, df_weather_test1, df_weather_test2], ignore_index=True)
    return df_train, df_test, df_weather


def weather_preprocess(weather):
    # missing time point
    # 2016-09-29 21
    # 2016-09-30 0
    # 2016-10-10 all day
    time_21 = weather.loc[(weather['date'] == '2016-09-29') & (weather['hour'] == 18), weather.columns]
    time_21.loc[:, 'hour'] = 21

    time_0 = weather.loc[(weather['date'] == '2016-09-30') & (weather['hour'] == 3), weather.columns]
    time_0.loc[:, 'hour'] = 0

    time_1010 = weather[weather['date'].isin(['2016-10-09', '2016-10-11'])].groupby(['hour'], as_index=False).mean()
    time_1010.loc[:, 'date'] = '2016-10-10'

    return weather.append([time_21, time_0, time_1010], ignore_index=True)



def holiday(date):
    # Weekend and National Day
    holiday_list = ['2016-09-24', '2016-09-25', '2016-10-01', '2016-10-02', '2016-10-03',
                    '2016-10-04', '2016-10-05', '2016-10-06', '2016-10-07', '2016-10-15',
                    '2016-10-16', '2016-10-22', '2016-10-23', '2016-10-29', '2016-10-30']
    if date in holiday_list:
        return 1
    else:
        return 0


def holiday_1day_before(date):
    # One day before holiday may have greater volume
    holiday_1day_before_list = ['2016-09-23', '2016-09-30', '2016-10-14', '2016-10-21', '2016-10-28']
    if date in holiday_1day_before_list:
        return 1
    else:
        return 0


def holiday_1day_after(date):
    holiday_1day_before_list = ['2016-09-26', '2016-10-08', '2016-10-17', '2016-10-24', '2016-10-31']
    if date in holiday_1day_before_list:
        return 1
    else:
        return 0


def week(date):
    # Predict cycle
    date = str(date)
    weeks = ['2016-09-26', '2016-10-03', '2016-10-10', '2016-10-17', '2016-10-24', '2016-10-31']
    if date <= weeks[0]:
        return 0
    elif date <= weeks[1]:
        return 1
    elif date <= weeks[2]:
        return 2
    elif date <= weeks[3]:
        return 3
    elif date <= weeks[4]:
        return 4
    elif date <= weeks[5]:
        return 5


def precipitation(x):
    # Rain or not
    if x > 0:
        return 1
    else:
        return 0


def temperature(x):
    if x < 20:
        return 1
    elif x < 30:
        return 2
    else:
        return 3


def wind_speed(x):
    if x < 3.3:
        return 0
    else:
        return 1


def direction_count(x):
    if x == 2:
        return 1
    else:
        return 2


def hour_section(x):
    # Day and night
    if x >= 7 & x <= 22:
        return 1
    else:
        return 0


def sample(df, cols):
    # Count Volume
    return df.groupby(cols, as_index=False)['minute'].count().rename(columns={'minute': 'volume'})


def slice_windows_sample(data, features, slice_point_array):
    df = []
    for slice_point in slice_point_array:
        hour = int(np.floor(slice_point/60))
        minute = slice_point - hour*60
        current_data = data.loc[(data['hour_minute'] >= slice_point) & (data['hour_minute'] < slice_point + 20), data.columns]
        current_data['hour'] = hour
        current_data['minute_point'] = minute
        current_data['slice_point'] = slice_point
        current_data = sample(current_data, features)
        print(slice_point, hour, minute)
        df.append(current_data)
    df = pd.concat(df, ignore_index=True)
    return df


def mape(y_true, y_pred):
    return np.abs((y_true - y_pred)/y_true).sum() / len(y_true)


def mae(y_true, y_pred):
    return np.abs(y_true - y_pred).sum() / len(y_true)


def print_mape(y_true, y_pred, alg_name):
    print(alg_name, ' MAPE:', mape(y_true, y_pred))
    print(alg_name, ' MAE:', mae(y_true, y_pred))


def lightgbm(train_sample, validation_sample, features, model_param):
    def evalmape(preds, train_data):
        labels = train_data.get_label()
        preds = np.power(log_base, preds) - log_bias
        return 'mape', np.abs((labels - preds) / labels).sum() / len(labels), False

    log_base = np.e
    log_bias = 1
    lgb_train = lgb.Dataset(train_sample[features], np.log(log_bias + train_sample['volume'])/np.log(log_base),
                            # weight=train_sample['weight']
                            )
    lgb_eval = lgb.Dataset(validation_sample[features], validation_sample['volume'], reference=lgb_train)
    params = {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'regression_l2',
        'metric': {'l2', 'l1'},
        'max_depth': model_param['depth'],
        'num_leaves': model_param['leaf'],
        'min_data_in_leaf': 20,
        'learning_rate': model_param['lr'],
        'feature_fraction': 1,
        'bagging_fraction': model_param['sample'],
        'bagging_freq': 1,
        'bagging_seed': model_param['seed']
    }

    gbm = lgb.train(params,
                    lgb_train,
                    num_boost_round=model_param['tree'],
                    valid_sets=lgb_eval,
                    feval=evalmape,
                    # learning_rates=lambda iter: 0.1 - 0.00002*iter,
                    # learning_rates=lambda iter: 0.1 * (0.999998 ** iter),
                    # categorical_feature=[0]
                    # early_stopping_rounds=0
                    )
    lightgbm_prob = gbm.predict(validation_sample[features])
    lightgbm_prob = np.power(log_base, lightgbm_prob) - log_bias
    # MAPE
    print('Feature importances:', list(gbm.feature_importance()))
    print_mape(validation_sample['volume'], lightgbm_prob, 'LIGHTGBM')
    return lightgbm_prob


def xgboost(train_sample, validation_sample, features, model_param):
    def evalmape(preds, dtrain):
        labels = dtrain.get_label()
        preds = np.power(log_base, preds) - 1
        # return a pair metric_name, result
        # since preds are margin(before logistic transformation, cutoff at 0)
        return 'mape', np.abs((labels - preds) / labels).sum() / len(labels)

    param = {'max_depth': model_param['depth'], 'eta': model_param['lr'], 'silent': 1, 'objective': 'reg:linear', 'booster': 'gbtree',
             'subsample': model_param['sample'],
             'seed':model_param['seed'],
             'colsample_bytree':1, 'min_child_weight':1, 'gamma':0}
    param['eval_metric'] = 'mae'
    num_round = model_param['tree']
    log_base = np.e
    plst = param.items()
    dtrain = xgb.DMatrix(train_sample[features], np.log1p(train_sample['volume'])/np.log(log_base))
    dtest = xgb.DMatrix(validation_sample[features], validation_sample['volume'])
    watchlist = [(dtest, 'eval'), (dtrain, 'train')]
    bst = xgb.train(plst, dtrain, num_round, watchlist, feval=evalmape)
    xgboost_prob = np.power(log_base, bst.predict(dtest)) - 1
    # MAPE
    print_mape(validation_sample['volume'], xgboost_prob, 'XGBOOST')
    return xgboost_prob


def gbrt(train_sample, validation_sample, features, model_param):
    log_base = np.e
    gbrt_est = GradientBoostingRegressor(n_estimators=model_param['tree'],
                                         learning_rate=model_param['lr'],
                                         max_depth=model_param['depth'],
                                         subsample=model_param['sample'],
                                         random_state=model_param['seed'],
                                         loss='ls').fit(
        train_sample[features], np.log1p(train_sample['volume']) / np.log(log_base))
    gbrt_prob = np.power(log_base, gbrt_est.predict(validation_sample[features])) - 1
    print_mape(validation_sample['volume'], gbrt_prob, 'GBRT')
    return gbrt_prob


def rf(train_sample, validation_sample, features, seed):
    log_base = np.e
    rf_est = RandomForestRegressor(n_estimators=500,
                                   criterion='mse',
                                   max_features=4,
                                   max_depth=None,
                                   bootstrap=True,
                                   min_samples_split=4,
                                   min_samples_leaf=1,
                                   min_weight_fraction_leaf=0,
                                   max_leaf_nodes=None,
                                   random_state=seed
                                   ).fit(
        train_sample[features], np.log1p(train_sample['volume']) / np.log(log_base))
    rf_prob = np.power(log_base, rf_est.predict(validation_sample[features])) - 1
    print_mape(validation_sample['volume'], rf_prob, 'RF')
    return rf_prob


def exrf(train_sample, validation_sample, features, seed):
    log_base = np.e
    exrf_est = ExtraTreesRegressor(n_estimators=1000,
                                   criterion='mse',
                                   max_features='auto',
                                   max_depth=None,
                                   bootstrap=True,
                                   min_samples_split=4,
                                   min_samples_leaf=1,
                                   min_weight_fraction_leaf=0,
                                   max_leaf_nodes=None,
                                   random_state=seed
                                       ).fit(
        train_sample[features], np.log1p(train_sample['volume']) / np.log(log_base))
    exrf_prob = np.power(log_base, exrf_est.predict(validation_sample[features])) - 1
    print_mape(validation_sample['volume'], exrf_prob, 'EXTRA-RF')
    return exrf_prob

def knn(train_sample, validation_sample, features, seed):
    log_base = np.e
    knn_est = KNeighborsRegressor(n_neighbors=1, weights='distance', algorithm='auto', leaf_size=30,
                                  p=1).fit(
        train_sample[features], np.log1p(train_sample['volume']) / np.log(log_base))
    knn_prob = np.power(log_base, knn_est.predict(validation_sample[features])) - 1
    print_mape(validation_sample['volume'], knn_prob, 'KNN')
    return knn_prob

def svm(train_sample, validation_sample, features):
    log_base = np.e
    svr_rbf = SVR(kernel='rbf', gamma='auto', coef0=0.0, tol=0.001, C=1, epsilon=0.1,
                  shrinking=True)
    svm_prob = svr_rbf.fit(train_sample[features], np.log1p(train_sample['volume'])/np.log(log_base)).predict(validation_sample[features])
    svm_prob = np.power(log_base, svm_prob) - 1
    print_mape(validation_sample['volume'], svm_prob, 'SVM')
    return svm_prob


def lr(train_sample, validation_sample, features):
    log_base = np.e
    lr_prob = LinearSVR(C=1, epsilon=0.1).fit(train_sample[features], np.log1p(train_sample['volume'])/np.log(log_base))\
        .predict(validation_sample[features])
    lr_prob = np.power(log_base, lr_prob) - 1
    print_mape(validation_sample['volume'], lr_prob, 'LR')
    return lr_prob

def one_hot(df, cols):
    for each in cols:
        dummies = pd.get_dummies(df[each], prefix=each, drop_first=False)
        df = pd.concat([df, dummies], axis=1)
    return df


def submit(result, result_path, predict_name):
    # Create a dictionary to caculate and store volume per time window
    volumes = {}  # key: time window value: dictionary
    for index, row in result.iterrows():
        month = int(row['month'])
        day = int(row['day'])
        hour = int(row['hour'])
        minute = int(row['minute_point'])
        tollgate_id = int(row['tollgate_id'])
        direction = int(row['direction'])
        volume = row[predict_name]

        # print pass_time
        start_time_window = datetime(2016, month, day, hour, minute, 0)

        if start_time_window not in volumes:
            volumes[start_time_window] = {}
        if tollgate_id not in volumes[start_time_window]:
            volumes[start_time_window][tollgate_id] = {}
        if direction not in volumes[start_time_window][tollgate_id]:
            volumes[start_time_window][tollgate_id][direction] = volume

    # format output for tollgate and direction per time window
    fw = open(result_path, 'w')
    fw.writelines(','.join(['"tollgate_id"', '"time_window"', '"direction"', '"volume"']) + '\n')
    time_windows = list(volumes.keys())
    time_windows.sort()
    for time_window_start in time_windows:
        time_window_end = time_window_start + timedelta(minutes=20)
        for tollgate_id in volumes[time_window_start]:
            for direction in volumes[time_window_start][tollgate_id]:
               out_line = ','.join(['"' + str(tollgate_id) + '"',
                                 '"[' + str(time_window_start) + ',' + str(time_window_end) + ')"',
                                 '"' + str(direction) + '"',
                                 '"' + str(volumes[time_window_start][tollgate_id][direction]) + '"',
                               ]) + '\n'
               fw.writelines(out_line)
    fw.close()


def weight(x):
    predict_point_array = [480, 500, 520, 540, 560, 580,
                           1020, 1040, 1060, 1080, 1100, 1120]
    if x in predict_point_array:
        return 2
    else:
        return 1


def train_filter(train_sample, filter_param):
    return train_sample.loc[(train_sample['minute_point'] % filter_param['slice_window'] == 0) &
                            (~((train_sample['month'] == filter_param['day_filter_0930'][0]) & (train_sample['day'] == filter_param['day_filter_0930'][1]))) &
                            (~((train_sample['month'] == filter_param['day_filter_1001'][0]) & (train_sample['day'] == filter_param['day_filter_1001'][1]))) &
                            (~((train_sample['tollgate_id'] == filter_param['tollgate_direction_filter'][0]) & (train_sample['direction'] == filter_param['tollgate_direction_filter'][1]))) &
                            (train_sample['slice_point'] >= filter_param['slice_point_min']) &
                            (train_sample['slice_point'] <= filter_param['slice_point_max']) &
                            (train_sample['volume'] >= filter_param['volume_min']) &
                            (train_sample['volume'] <= filter_param['volume_max'])
                            , train_sample.columns]


def train_filter_percent(train_sample, percent):
    def lightgbm_filter(train_sample, validation_sample, features, seed):
        def evalmape(preds, train_data):
            labels = train_data.get_label()
            preds = np.power(log_base, preds) - log_bias
            return 'mape', np.abs((labels - preds) / labels).sum() / len(labels), False

        log_base = np.e
        log_bias = 1
        lgb_train = lgb.Dataset(train_sample[features],
                                np.log(log_bias + train_sample['volume']) / np.log(log_base),
                                # weight=train_sample['weight']
                                )
        lgb_eval = lgb.Dataset(validation_sample[features], validation_sample['volume'], reference=lgb_train)
        params = {
            'task': 'train',
            'boosting_type': 'gbdt',
            'objective': 'regression_l2',
            'metric': {'l2', 'l1'},
            'max_depth': 3,
            'num_leaves': 400,
            'min_data_in_leaf': 20,
            'learning_rate': 0.1,
            'feature_fraction': 1,
            'bagging_fraction': 1,
            'bagging_freq': 1,
            'bagging_seed': seed
        }

        gbm = lgb.train(params,
                        lgb_train,
                        num_boost_round=500,
                        valid_sets=lgb_eval,
                        feval=evalmape,
                        )
        lightgbm_prob = gbm.predict(validation_sample[features])
        lightgbm_prob = np.power(log_base, lightgbm_prob) - log_bias
        # MAPE
        print_mape(validation_sample['volume'], lightgbm_prob, 'LIGHTGBM')
        print('Feature importances:', list(gbm.feature_importance()))

        return lightgbm_prob
        # return mape(validation_sample['volume'], lightgbm_prob)


    features = list(train_sample.columns.values)
    features.remove('volume')
    print('xxxxxxx------------')
    print(features)
    # features.remove('hour')
    # features.remove('minute_point')

    train_predict = lightgbm_filter(train_sample, train_sample, features, 0)
    train_sample['error_abs'] = np.abs(train_predict - train_sample['volume'])
    train_sample.to_csv("sample_final/xx.csv", index=False)

    xx = train_sample.loc[train_sample['error_abs'] < train_sample['error_abs'].quantile(percent),
                          ['month', 'day_of_week', 'day', 'holiday', 'hour', 'minute_point', 'tollgate_id', 'direction',
                           'slice_point', 'volume']]
    print(train_sample.shape)
    print(xx.shape)
    print('--------------------------')
    return xx

def model(model_name, train_sample, validation_sample, features, day_range, filter_param, model_param):
    hour_range = [8, 17]
    train_sample = train_filter(train_sample, filter_param)
    df = []
    for curr_day in day_range:
        for curr_hour in hour_range:
            curr_train_sample = train_sample.loc[~((train_sample['month'] == 10)
                                                 & (train_sample['day'] > curr_day)
                                                 | (train_sample['month'] == 10)
                                                 & (train_sample['day'] == curr_day)
                                                 & (train_sample['hour'] > curr_hour)
                                                   ), train_sample.columns]
            curr_validation_sample = validation_sample.loc[((validation_sample['month'] == 10)
                                                 & (validation_sample['day'] == curr_day)
                                                 & (validation_sample['hour'] >= curr_hour)
                                                 & (validation_sample['hour'] < (curr_hour+2))), validation_sample.columns]
            print(curr_day, "-", curr_hour, curr_train_sample.shape)
            print(curr_day, "-", curr_hour, curr_validation_sample.shape)
            if curr_validation_sample.shape[0] > 0:
                if model_name == "lightgbm":
                    curr_predict = lightgbm(curr_train_sample, curr_validation_sample, features, model_param)
                elif model_name == "xgboost":
                    curr_predict = xgboost(curr_train_sample, curr_validation_sample, features, model_param)
                else:
                    curr_predict = gbrt(curr_train_sample, curr_validation_sample, features, model_param)
                curr_validation_sample[model_name] = curr_predict

            df.append(curr_validation_sample)

    df = pd.concat(df, ignore_index=True)
    print_mape(df['volume'], df[model_name], model_name)
    return df


def main():
    # Data Merge & Preprocess
    volume_train, volume_test_pre, weather = load_data(VOLUME_TABLE6_TRAIN, VOLUME_TABLE6_TRAIN2, VOLUME_TABLE6_TEST2,
                                                   WEATHER_TABLE7_UPDATE, WEATHER_TABLE7_test1, WEATHER_TABLE7_test2)
    weather = weather_preprocess(weather)
    print("weather:", weather.shape)

    # Slice Window Sample
    volume_test_point = volume_test_pre.loc[:, volume_test_pre.columns]
    volume_test_point['time'] = pd.to_datetime(volume_test_pre['time']) + pd.DateOffset(hours=2)
    volume = pd.concat([volume_train, volume_test_pre, volume_test_point], ignore_index=True)

    # Time feature
    volume['time'] = pd.to_datetime(volume['time'])
    volume['date'] = volume['time'].apply(lambda x: x.strftime('%Y-%m-%d'))
    volume['minute'] = volume['time'].apply(lambda x: x.minute)
    volume['month'] = volume['time'].apply(lambda x: x.month)
    volume['day_of_week'] = volume['time'].apply(lambda x: x.dayofweek)
    volume['day_of_year'] = volume['time'].apply(lambda x: x.dayofyear)
    volume['week_of_year'] = volume['time'].apply(lambda x: x.weekofyear)

    volume['day'] = volume['time'].apply(lambda x: x.day)
    volume['hour'] = volume['time'].apply(lambda x: x.hour)
    volume['hour_minute'] = volume['hour'] * 60 + volume['minute']
    volume['holiday'] = volume['date'].apply(holiday)

    volume['holiday_1day_before'] = volume['date'].apply(holiday_1day_before)
    volume['holiday_1day_after'] = volume['date'].apply(holiday_1day_after)
    volume['direction_count'] = volume['tollgate_id'].apply(direction_count)

    volume['week'] = volume['time'].apply(week)
    volume['weather_hour_map'] = volume['time'].apply(lambda x: int(np.floor(x.hour / 3) * 3))
    volume['hour_section'] = volume['hour'].apply(hour_section)

    # Join weather feature
    weather = weather.rename(columns={'hour': 'weather_hour_map'})
    volume = pd.merge(volume, weather,
                      on=['date', 'weather_hour_map'],
                      how='inner')
    volume['precipitation'] = volume['precipitation'].apply(precipitation)
    volume['temperature'] = volume['temperature'].apply(temperature)
    volume['wind_speed'] = volume['wind_speed'].apply(wind_speed)

    # Split Offline and Online DataSet
    # Offline Validation
    # Online Submit
    off_date_begin, off_date_end = '2016-09-19', '2016-10-17'
    on_date_begin, on_date_end = '2016-09-19', '2016-10-24'
    evaluate_period = 7
    off_evaluate_begin = (datetime.strptime(off_date_end, '%Y-%m-%d') + timedelta(days=1)).strftime('%Y-%m-%d')
    off_evaluate_end = (datetime.strptime(off_date_end, '%Y-%m-%d') + timedelta(days=evaluate_period)).strftime('%Y-%m-%d')
    on_evaluate_begin = (datetime.strptime(on_date_end, '%Y-%m-%d') + timedelta(days=1)).strftime('%Y-%m-%d')
    on_evaluate_end = (datetime.strptime(on_date_end, '%Y-%m-%d') + timedelta(days=evaluate_period)).strftime('%Y-%m-%d')

    off_train = volume.loc[(volume['date'] >= off_date_begin) & (volume['date'] <= off_date_end), volume.columns]
    validation_pre = volume.loc[(volume['date'] >= off_evaluate_begin) & (volume['date'] <= off_evaluate_end) &
                                (volume['hour'].isin([6, 7, 15, 16])), volume.columns]
    validation_point = volume.loc[(volume['date'] >= off_evaluate_begin) & (volume['date'] <= off_evaluate_end) &
                                  (volume['hour'].isin([8, 9, 17, 18])), volume.columns]

    on_train = volume.loc[(volume['date'] >= on_date_begin) & (volume['date'] <= on_date_end), volume.columns]
    test_pre = volume.loc[(volume['date'] >= on_evaluate_begin) & (volume['date'] <= on_evaluate_end) &
                          (volume['hour'].isin([6, 7, 15, 16])), volume.columns]
    test_point = volume.loc[(volume['date'] >= on_evaluate_begin) & (volume['date'] <= on_evaluate_end) &
                            (volume['hour'].isin([8, 9, 17, 18])), volume.columns]

    # Slice Windows
    features = [
        'month',
        'day_of_week',
        'day',
        'holiday',
        'hour',
        'minute_point',
        'tollgate_id',
        'direction',
        'slice_point',
    ]
    # point
    train_point_array = np.arange(0, 1421, 1)
    off_train_sample = slice_windows_sample(off_train, features, train_point_array)
    on_train_sample = slice_windows_sample(on_train, features, train_point_array)

    # pre
    train_pre_array = np.arange(360, 461, 1).tolist() + np.arange(900, 1001, 1).tolist()
    validation_pre_sample = slice_windows_sample(validation_pre, features, train_pre_array)
    test_pre_sample = slice_windows_sample(test_pre, features, train_pre_array)

    # 8*60 + 5*20
    # 17*60 + 5*20
    predict_point_array = [480, 500, 520, 540, 560, 580,
                           1020, 1040, 1060, 1080, 1100, 1120]
    validation_sample = slice_windows_sample(validation_point, features, predict_point_array)
    test_sample = slice_windows_sample(test_point, features, predict_point_array)

    off_train_sample = pd.concat([off_train_sample,
                                  validation_pre_sample,
                                  # validation_pre_sample.sample(frac=0.5, replace=False, random_state=0),
                                  # off_train_sample[off_train_sample['slice_point'].isin(predict_point_array)].sample(frac=0.2, replace=False),
                                  ], ignore_index=True)
    on_train_sample = pd.concat([on_train_sample, test_pre_sample], ignore_index=True)

    print('off_train_sample', off_train_sample.shape)
    print('on_train_sample', on_train_sample.shape)
    print('validation_sample', validation_sample.shape)
    print('test_sample', test_sample.shape)

    # Model
    # off_train_sample, validation_sample = load_data(OFF_TRAIN_SAMPLE, VALIDATION_SAMPLE)
    # on_train_sample, test_sample = load_data(ON_TRAIN_SAMPLE, TEST_SAMPLE)
    #off_train_sample, validation_sample = load_data(ON_TRAIN_SAMPLE, TEST_SAMPLE)
    # day_range = np.arange(18, 25, 1)
    off_train_sample = on_train_sample
    validation_sample = test_sample
    day_range = np.arange(25, 32, 1)
    features = list(off_train_sample.columns.values)
    features.remove('volume')
    print(features)

    ## model all
    filter_param = {'slice_window': 1, 'day_filter_0930': [9, 30], 'day_filter_1001': [10, 1],
                    'tollgate_direction_filter': [0, 0],
                    'slice_point_min': 240, 'slice_point_max': 1420, 'volume_min': 1, 'volume_max': 1000}
    model_param = {'lr': 0.1, 'depth': 6, 'tree': 500, 'leaf': 400, 'sample': 0.9, 'seed': 3}
    lightgbm_result = model("lightgbm", off_train_sample, validation_sample, features, day_range, filter_param, model_param)

    filter_param = {'slice_window': 10, 'day_filter_0930': [9, 30], 'day_filter_1001': [10, 1],
                    'tollgate_direction_filter': [0, 0],
                    'slice_point_min': 240, 'slice_point_max': 1420, 'volume_min': 1, 'volume_max': 1000}
    model_param = {'lr': 0.11, 'depth': 5, 'tree': 500, 'leaf': 400, 'sample': 0.9, 'seed': 0}
    xgboost_result = model("xgboost", off_train_sample, validation_sample, features, day_range, filter_param, model_param)

    filter_param = {'slice_window': 5, 'day_filter_0930': [9, 30], 'day_filter_1001': [10, 1],
                    'tollgate_direction_filter': [0, 0],
                    'slice_point_min': 240, 'slice_point_max': 1420, 'volume_min': 1, 'volume_max': 1000}
    model_param = {'lr': 0.11, 'depth': 5, 'tree': 500, 'leaf': 400, 'sample': 0.9, 'seed': 0}
    gbrt_result = model("gbrt", off_train_sample, validation_sample, features, day_range, filter_param, model_param)

    ensemble_3model = lightgbm_result.merge(xgboost_result, on=features).merge(gbrt_result, on=features)
    ensemble_3model['predict1'] = (ensemble_3model['lightgbm'] + ensemble_3model['xgboost'] + ensemble_3model['gbrt']) / 3

    # model separate tollgate direction lightgbm
    separate_sample = validation_sample.loc[
        (validation_sample['tollgate_id'] == 1) & (validation_sample['direction'] == 0), validation_sample.columns]
    filter_param = {'slice_window': 5, 'day_filter_0930': [0, 0], 'day_filter_1001': [0, 0],
                    'tollgate_direction_filter': [3, 0],
                    'slice_point_min': 0, 'slice_point_max': 1420, 'volume_min': 1, 'volume_max': 1000}
    model_param = {'lr': 0.1, 'depth': 6, 'tree': 300, 'leaf': 400, 'sample': 0.9, 'seed': 3}
    lightgbm_separate_10 = model("lightgbm", off_train_sample, separate_sample, features, day_range, filter_param,
                                 model_param)

    separate_sample = validation_sample.loc[
        (validation_sample['tollgate_id'] == 1) & (validation_sample['direction'] == 1), validation_sample.columns]
    filter_param = {'slice_window': 5, 'day_filter_0930': [0, 0], 'day_filter_1001': [0, 0],
                    'tollgate_direction_filter': [1, 0],
                    'slice_point_min': 360, 'slice_point_max': 1420, 'volume_min': 1, 'volume_max': 1000}
    model_param = {'lr': 0.1, 'depth': 6, 'tree': 300, 'leaf': 400, 'sample': 0.9, 'seed': 3}
    lightgbm_separate_11 = model("lightgbm", off_train_sample, separate_sample, features, day_range, filter_param,
                                 model_param)

    separate_sample = validation_sample.loc[
        (validation_sample['tollgate_id'] == 2) & (validation_sample['direction'] == 0), validation_sample.columns]
    filter_param = {'slice_window': 5, 'day_filter_0930': [0, 0], 'day_filter_1001': [0, 0],
                    'tollgate_direction_filter': [0, 0],
                    'slice_point_min': 0, 'slice_point_max': 1420, 'volume_min': 1, 'volume_max': 1000}
    model_param = {'lr': 0.1, 'depth': 6, 'tree': 500, 'leaf': 400, 'sample': 0.9, 'seed': 3}
    lightgbm_separate_20 = model("lightgbm", off_train_sample, separate_sample, features, day_range, filter_param,
                                 model_param)

    separate_sample = validation_sample.loc[
        (validation_sample['tollgate_id'] == 3) & (validation_sample['direction'] == 0), validation_sample.columns]
    filter_param = {'slice_window': 1, 'day_filter_0930': [9, 30], 'day_filter_1001': [10, 1],
                    'tollgate_direction_filter': [0, 0],
                    'slice_point_min': 240, 'slice_point_max': 1420, 'volume_min': 1, 'volume_max': 1000}
    model_param = {'lr': 0.1, 'depth': 6, 'tree': 500, 'leaf': 400, 'sample': 0.9, 'seed': 3}
    lightgbm_separate_30 = model("lightgbm", off_train_sample, separate_sample, features, day_range, filter_param,
                                 model_param)

    separate_sample = validation_sample.loc[
        (validation_sample['tollgate_id'] == 3) & (validation_sample['direction'] == 1), validation_sample.columns]
    filter_param = {'slice_window': 5, 'day_filter_0930': [0, 0], 'day_filter_1001': [0, 0],
                    'tollgate_direction_filter': [1, 1],
                    'slice_point_min': 0, 'slice_point_max': 1420, 'volume_min': 1, 'volume_max': 1000}
    model_param = {'lr': 0.1, 'depth': 6, 'tree': 500, 'leaf': 400, 'sample': 0.9, 'seed': 3}
    lightgbm_separate_31 = model("lightgbm", off_train_sample, separate_sample, features, day_range, filter_param,
                                 model_param)

    lightgbm_separate_tollgate_direction = pd.concat([lightgbm_separate_10, lightgbm_separate_11, lightgbm_separate_20,
                                                      lightgbm_separate_30, lightgbm_separate_31], ignore_index=True)
    lightgbm_separate_tollgate_direction['predict'] = lightgbm_separate_tollgate_direction['lightgbm']

    # model separate tollgate direction xgboost
    separate_sample = validation_sample.loc[
        (validation_sample['tollgate_id'] == 1) & (validation_sample['direction'] == 0), validation_sample.columns]
    filter_param = {'slice_window': 5, 'day_filter_0930': [0, 0], 'day_filter_1001': [0, 0],
                    'tollgate_direction_filter': [0, 0],
                    'slice_point_min': 240, 'slice_point_max': 1420, 'volume_min': 1, 'volume_max': 1000}
    model_param = {'lr': 0.12, 'depth': 5, 'tree': 500, 'leaf': 400, 'sample': 0.9, 'seed': 0}
    xgboost_separate_10 = model("xgboost", off_train_sample, separate_sample, features, day_range, filter_param,
                                model_param)

    separate_sample = validation_sample.loc[
        (validation_sample['tollgate_id'] == 1) & (validation_sample['direction'] == 1), validation_sample.columns]
    filter_param = {'slice_window': 10, 'day_filter_0930': [0, 0], 'day_filter_1001': [0, 0],
                    'tollgate_direction_filter': [1, 0],
                    'slice_point_min': 120, 'slice_point_max': 1420, 'volume_min': 1, 'volume_max': 1000}
    model_param = {'lr': 0.11, 'depth': 5, 'tree': 500, 'leaf': 400, 'sample': 0.9, 'seed': 0}
    xgboost_separate_11 = model("xgboost", off_train_sample, separate_sample, features, day_range, filter_param,
                                model_param)

    separate_sample = validation_sample.loc[
        (validation_sample['tollgate_id'] == 2) & (validation_sample['direction'] == 0), validation_sample.columns]
    filter_param = {'slice_window': 20, 'day_filter_0930': [0, 0], 'day_filter_1001': [0, 0],
                    'tollgate_direction_filter': [0, 0],
                    'slice_point_min': 240, 'slice_point_max': 1420, 'volume_min': 1, 'volume_max': 1000}
    model_param = {'lr': 0.11, 'depth': 5, 'tree': 500, 'leaf': 400, 'sample': 0.9, 'seed': 0}
    xgboost_separate_20 = model("xgboost", off_train_sample, separate_sample, features, day_range, filter_param,
                                model_param)

    separate_sample = validation_sample.loc[
        (validation_sample['tollgate_id'] == 3) & (validation_sample['direction'] == 0), validation_sample.columns]
    filter_param = {'slice_window': 10, 'day_filter_0930': [0, 0], 'day_filter_1001': [0, 0],
                    'tollgate_direction_filter': [0, 0],
                    'slice_point_min': 180, 'slice_point_max': 1420, 'volume_min': 1, 'volume_max': 1000}
    model_param = {'lr': 0.1, 'depth': 5, 'tree': 500, 'leaf': 400, 'sample': 0.9, 'seed': 0}
    xgboost_separate_30 = model("xgboost", off_train_sample, separate_sample, features, day_range, filter_param,
                                model_param)

    separate_sample = validation_sample.loc[
        (validation_sample['tollgate_id'] == 3) & (validation_sample['direction'] == 1), validation_sample.columns]
    filter_param = {'slice_window': 5, 'day_filter_0930': [0, 0], 'day_filter_1001': [0, 0],
                    'tollgate_direction_filter': [1, 1],
                    'slice_point_min': 60, 'slice_point_max': 1420, 'volume_min': 1, 'volume_max': 1000}
    model_param = {'lr': 0.11, 'depth': 5, 'tree': 500, 'leaf': 400, 'sample': 0.9, 'seed': 0}
    xgboost_separate_31 = model("xgboost", off_train_sample, separate_sample, features, day_range, filter_param,
                                model_param)

    xgboost_separate_tollgate_direction = pd.concat([xgboost_separate_10, xgboost_separate_11, xgboost_separate_20,
                                                     xgboost_separate_30, xgboost_separate_31], ignore_index=True)

    # # model separate tollgate direction gbrt
    separate_sample = validation_sample.loc[
        (validation_sample['tollgate_id'] == 1) & (validation_sample['direction'] == 0), validation_sample.columns]
    filter_param = {'slice_window': 5, 'day_filter_0930': [0, 0], 'day_filter_1001': [0, 0],
                    'tollgate_direction_filter': [0, 0],
                    'slice_point_min': 240, 'slice_point_max': 1420, 'volume_min': 1, 'volume_max': 1000}
    model_param = {'lr': 0.08, 'depth': 5, 'tree': 500, 'leaf': 400, 'sample': 0.9, 'seed': 0}
    gbrt_separate_10 = model("gbrt", off_train_sample, separate_sample, features, day_range, filter_param, model_param)

    separate_sample = validation_sample.loc[
        (validation_sample['tollgate_id'] == 1) & (validation_sample['direction'] == 1), validation_sample.columns]
    filter_param = {'slice_window': 20, 'day_filter_0930': [0, 0], 'day_filter_1001': [0, 0],
                    'tollgate_direction_filter': [1, 0],
                    'slice_point_min': 240, 'slice_point_max': 1420, 'volume_min': 1, 'volume_max': 1000}
    model_param = {'lr': 0.11, 'depth': 5, 'tree': 500, 'leaf': 400, 'sample': 0.9, 'seed': 0}
    gbrt_separate_11 = model("gbrt", off_train_sample, separate_sample, features, day_range, filter_param, model_param)

    separate_sample = validation_sample.loc[
        (validation_sample['tollgate_id'] == 2) & (validation_sample['direction'] == 0), validation_sample.columns]
    filter_param = {'slice_window': 5, 'day_filter_0930': [0, 0], 'day_filter_1001': [0, 0],
                    'tollgate_direction_filter': [1, 0],
                    'slice_point_min': 180, 'slice_point_max': 1420, 'volume_min': 1, 'volume_max': 1000}
    model_param = {'lr': 0.11, 'depth': 5, 'tree': 500, 'leaf': 400, 'sample': 0.9, 'seed': 0}
    gbrt_separate_20 = model("gbrt", off_train_sample, separate_sample, features, day_range, filter_param, model_param)

    separate_sample = validation_sample.loc[
        (validation_sample['tollgate_id'] == 3) & (validation_sample['direction'] == 0), validation_sample.columns]
    filter_param = {'slice_window': 5, 'day_filter_0930': [0, 0], 'day_filter_1001': [0, 0],
                    'tollgate_direction_filter': [0, 0],
                    'slice_point_min': 120, 'slice_point_max': 1420, 'volume_min': 1, 'volume_max': 1000}
    model_param = {'lr': 0.11, 'depth': 5, 'tree': 500, 'leaf': 400, 'sample': 0.9, 'seed': 0}
    gbrt_separate_30 = model("gbrt", off_train_sample, separate_sample, features, day_range, filter_param, model_param)

    separate_sample = validation_sample.loc[
        (validation_sample['tollgate_id'] == 3) & (validation_sample['direction'] == 1), validation_sample.columns]
    filter_param = {'slice_window': 5, 'day_filter_0930': [0, 0], 'day_filter_1001': [0, 0],
                    'tollgate_direction_filter': [1, 1],
                    'slice_point_min': 240, 'slice_point_max': 1420, 'volume_min': 1, 'volume_max': 1000}
    model_param = {'lr': 0.14, 'depth': 5, 'tree': 500, 'leaf': 400, 'sample': 0.9, 'seed': 0}
    gbrt_separate_31 = model("gbrt", off_train_sample, separate_sample, features, day_range, filter_param, model_param)

    gbrt_separate_tollgate_direction = pd.concat([gbrt_separate_10, gbrt_separate_11, gbrt_separate_20,
                                                  gbrt_separate_30, gbrt_separate_31], ignore_index=True)

    ## model separate tollgate direction
    ensemble_separate_tollgate_direction = lightgbm_separate_tollgate_direction.merge(
        xgboost_separate_tollgate_direction, on=features).merge(
        gbrt_separate_tollgate_direction, on=features)
    ensemble_separate_tollgate_direction['predict2'] = (ensemble_separate_tollgate_direction['lightgbm']
                                                        + ensemble_separate_tollgate_direction['xgboost']
                                                        + ensemble_separate_tollgate_direction['gbrt']) / 3

    # model separate morning afternoon lightgbm
    separate_sample = validation_sample.loc[(validation_sample['hour'] < 12), validation_sample.columns]
    filter_param = {'slice_window': 5, 'day_filter_0930': [0, 0], 'day_filter_1001': [0, 0],
                    'tollgate_direction_filter': [0, 0],
                    'slice_point_min': 0, 'slice_point_max': 1420, 'volume_min': 1, 'volume_max': 1000}
    model_param = {'lr': 0.1, 'depth': 6, 'tree': 500, 'leaf': 32, 'sample': 0.9, 'seed': 3}
    lightgbm_separate_morning = model("lightgbm", off_train_sample, separate_sample, features, day_range, filter_param,
                                      model_param)

    separate_sample = validation_sample.loc[(validation_sample['hour'] >= 12), validation_sample.columns]
    filter_param = {'slice_window': 1, 'day_filter_0930': [0, 0], 'day_filter_1001': [0, 0],
                    'tollgate_direction_filter': [0, 0],
                    'slice_point_min': 420, 'slice_point_max': 1420, 'volume_min': 1, 'volume_max': 1000}
    model_param = {'lr': 0.12, 'depth': 5, 'tree': 500, 'leaf': 400, 'sample': 0.9, 'seed': 3}
    lightgbm_separate_afternoon = model("lightgbm", off_train_sample, separate_sample, features, day_range, filter_param,
                                        model_param)

    lightgbm_separate_morning_afternoon = pd.concat([lightgbm_separate_morning,
                                                     lightgbm_separate_afternoon], ignore_index=True)

    # model separate morning afternoon xgboost
    separate_sample = validation_sample.loc[(validation_sample['hour'] < 12), validation_sample.columns]
    filter_param = {'slice_window': 10, 'day_filter_0930': [0, 0], 'day_filter_1001': [0, 0],
                    'tollgate_direction_filter': [0, 0],
                    'slice_point_min': 240, 'slice_point_max': 1420, 'volume_min': 1, 'volume_max': 1000}
    model_param = {'lr': 0.1, 'depth': 6, 'tree': 300, 'leaf': 400, 'sample': 0.9, 'seed': 0}
    xgboost_separate_morning = model("xgboost", off_train_sample, separate_sample, features, day_range, filter_param,
                                     model_param)

    separate_sample = validation_sample.loc[(validation_sample['hour'] >= 12), validation_sample.columns]
    filter_param = {'slice_window': 5, 'day_filter_0930': [0, 0], 'day_filter_1001': [0, 0],
                    'tollgate_direction_filter': [0, 0],
                    'slice_point_min': 300, 'slice_point_max': 1420, 'volume_min': 1, 'volume_max': 1000}
    model_param = {'lr': 0.11, 'depth': 5, 'tree': 700, 'leaf': 400, 'sample': 0.9, 'seed': 0}
    xgboost_separate_afternoon = model("xgboost", off_train_sample, separate_sample, features, day_range, filter_param,
                                       model_param)

    xgboost_separate_morning_afternoon = pd.concat([xgboost_separate_morning,
                                                    xgboost_separate_afternoon], ignore_index=True)

    # model separate morning afternoon gbrt
    separate_sample = validation_sample.loc[(validation_sample['hour'] < 12), validation_sample.columns]
    filter_param = {'slice_window': 5, 'day_filter_0930': [0, 0], 'day_filter_1001': [0, 0],
                    'tollgate_direction_filter': [0, 0],
                    'slice_point_min': 60, 'slice_point_max': 1420, 'volume_min': 1, 'volume_max': 1000}
    model_param = {'lr': 0.1, 'depth': 5, 'tree': 500, 'leaf': 400, 'sample': 0.9, 'seed': 0}
    gbrt_separate_morning = model("gbrt", off_train_sample, separate_sample, features, day_range, filter_param, model_param)

    separate_sample = validation_sample.loc[(validation_sample['hour'] >= 12), validation_sample.columns]
    filter_param = {'slice_window': 10, 'day_filter_0930': [0, 0], 'day_filter_1001': [0, 0],
                    'tollgate_direction_filter': [0, 0],
                    'slice_point_min': 300, 'slice_point_max': 1420, 'volume_min': 1, 'volume_max': 1000}
    model_param = {'lr': 0.11, 'depth': 5, 'tree': 600, 'leaf': 400, 'sample': 0.9, 'seed': 0}
    gbrt_separate_afternoon = model("gbrt", off_train_sample, separate_sample, features, day_range, filter_param,
                                    model_param)

    gbrt_separate_morning_afternoon = pd.concat([gbrt_separate_morning,
                                                 gbrt_separate_afternoon], ignore_index=True)

    ## model separate morning afternoon
    ensemble_separate_morning_afternoon = lightgbm_separate_morning_afternoon.merge(
        xgboost_separate_morning_afternoon, on=features).merge(
        gbrt_separate_morning_afternoon, on=features)
    ensemble_separate_morning_afternoon['predict3'] = (ensemble_separate_morning_afternoon['lightgbm']
                                                       + ensemble_separate_morning_afternoon['xgboost']
                                                       + ensemble_separate_morning_afternoon['gbrt']) / 3

    ### model final
    ensemble_3ensemble = ensemble_3model.merge(ensemble_separate_tollgate_direction, on=features).merge(
        ensemble_separate_morning_afternoon, on=features)
    ensemble_3ensemble['predict'] = (ensemble_3ensemble['predict1'] +
                                     ensemble_3ensemble['predict2'] +
                                     ensemble_3ensemble['predict3']) / 3

    print_mape(ensemble_3ensemble['volume'], ensemble_3ensemble['predict'], "ensemble_3ensemble")

    submit(ensemble_3model, "../../answer/predict_rice_1.csv", "predict1")
    submit(ensemble_separate_tollgate_direction, "../../answer/predict_rice_2.csv", "predict2")
    submit(ensemble_separate_morning_afternoon, "../../answer/predict_rice_3.csv", "predict3")
    submit(ensemble_3ensemble, "../../answer/predict_rice_final.csv", "predict")

if __name__ == '__main__':
    main()
